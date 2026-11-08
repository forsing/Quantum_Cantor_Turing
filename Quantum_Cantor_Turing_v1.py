# Quantum_Cantor_Turing

"""
Quantum_Cantor_Turing_V1.py

Python 3.11
Qiskit 1.4.4

OSNOVNA IDEJA
-------------
Ogranicena banka programa predstavlja hipoteze o prelazima kola.

Konstruktivna dijagonalna dopuna:
    za postojece deterministicke programe bira istorijske
    slucajeve u kojima njihov izlaz nije bio stvarni cilj;

    konstruiše novi program koji na tim razlicitim ulazima
    daje stvarne istorijske ciljeve;

    proverava da se novi program razlikuje od svakog
    obuhvacenog programa na njegovom izabranom slucaju.

Ovo je konacna prediktivna adaptacija dijagonalne konstrukcije.
Ne resava opsti problem zaustavljanja.

PROGRAMI
--------
AFFINE:
    cita prethodno ili pretprethodno kolo;
    primenjuje bijekciju brojeva modulo 39.

DIAGONAL:
    ograniceno grananje po poznatim istorijskim bitovima;
    primenjuje naucenu permutaciju brojeva.

LITERAL:
    24-bitna adresa legalne kombinacije;
    kanonsko kombinatorno dekodiranje.

Svaki kratki program uvek daje sedam razlicitih brojeva.
Literalni program adresira svih C(39,7) kombinacija.
Nema dopunjavanja ili popravljanja duplikata.

KVANTNI MODEL
-------------
Sest programskih kubita:
    istorijski uslovljene RY rotacije;
    CNOT lanac;
    amplitudna raspodela nad aktivnim programima.

Dvadeset cetiri literalna kubita:
    istorijski uslovljene RY rotacije;
    normalizacija na validne adrese kombinacija.

Qiskit racuna/proverava:
    programsko stanje;
    literalne jednokubitne amplitude;
    sestokubitne permutacione izlazne grane.

Sedam izlaznih registara koristi po sest kubita.
Simulacija je po programskim granama i faktorisanim literalnim
kubitima. Gust vektor celog sistema se ne pravi.

Kompilacija programa, ograniceni interpreter, kombinatorno
dekodiranje adrese i optimizacija su klasicni delovi.

Model nije klasicni regresor predstavljen kao kvantni izlaz:
ucene velicine su parametri kvantnih pripremnih kapija.

RASPodela
---------
    P(S | H) =
        q_literal(H) * P_literal(rank(S) | H)
        + sum_j q_j(H) * indicator(P_j(H) == S).

Mase programa koji daju isti skup se sabiraju.

Tacni globalni MAP nalazi se medju:
    literalnim MAP izlazom;
    svim izlazima kratkih programa.

Izvan programskih izlaza ostaje samo literalna komponenta,
pa druge kombinacije ne mogu imati vecu masu od njenog MAP-a.

UCENJE
------
Regresija raspodele preko NLL celih kombinacija.
Analiticki gradijent.
L-BFGS-B optimizuje kvantne parametre.

Dijagonalni programi i skaliranje uce se samo na treningu.
Hronoloska validacija bira broj dijagonalnih dopuna i L2.
Test je odvojen.
Zavrsni model koristi sve istorijske prelaze.

Bez klasifikatora, slucajnih kombinacija, dodatog suma,
pretpostavljene normalne raspodele i kozmetickih filtera.

DIAGNOSTICS=False daje kratke ispise.
"""

import csv
import hashlib
import itertools
import math
import random
from pathlib import Path

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from scipy.special import logsumexp


random.seed(39)
np.random.seed(39)


FILES = (
    Path("/data/loto7_4716_k89.csv"),
    Path("/data/loto7_4716_k89_loto_2980.csv"),
    Path("/data/loto7_4716_k89_loto_plus_1736.csv"),
)

DIAGNOSTICS = False

N = 39
K = 7
TOTAL_COMBINATIONS = math.comb(N, K)

PROGRAM_QUBITS = 6
PROGRAM_DIM = 1 << PROGRAM_QUBITS

LITERAL_QUBITS = (TOTAL_COMBINATIONS - 1).bit_length()
GATE_COUNT = PROGRAM_QUBITS + LITERAL_QUBITS

OUTPUT_QUBITS = 6
OUTPUT_DIM = 1 << OUTPUT_QUBITS

MAX_PROGRAM_STEPS = 128
DIAGONAL_LEVELS = (1, 3)
REGULARIZATIONS = (0.01, 0.1)

TRAIN_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85

BATCH_SIZE = 128
RESTARTS = 2
MAX_ITERATIONS = 160
GRADIENT_TOLERANCE = 1e-5

BASE_MULTIPLIERS = (1, 2, 5)
BASE_OFFSETS = (0, 1, 13, 26)
BASE_LAGS = (1, 2)

BINOMIAL = np.array([
    [
        math.comb(n, k) if k <= n else 0
        for k in range(K + 1)
    ]
    for n in range(N + 1)
], dtype=np.int64)

PROGRAM_BASIS = np.arange(PROGRAM_DIM, dtype=np.int64)
PROGRAM_RAW_BITS = (
    PROGRAM_BASIS[:, None]
    >> np.arange(PROGRAM_QUBITS, dtype=np.int64)[None, :]
) & 1

PROGRAM_PERMUTATION = PROGRAM_BASIS.copy()

for control in range(PROGRAM_QUBITS - 1):
    PROGRAM_PERMUTATION ^= (
        ((PROGRAM_PERMUTATION >> control) & 1)
        << (control + 1)
    )

PROGRAM_INVERSE = np.argsort(PROGRAM_PERMUTATION)


class DistributionError(ArithmeticError):
    pass


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def detail(*values):
    if DIAGNOSTICS:
        print(*values, flush=True)


def load_csv(path):
    raw = path.read_bytes()
    rows = []

    for line_number, row in enumerate(
        csv.reader(raw.decode("utf-8-sig").splitlines()),
        start=1,
    ):
        if not row or all(not cell.strip() for cell in row):
            continue

        require(
            len(row) == K,
            f"{path.name}, red {line_number}: potrebno je sedam kolona.",
        )

        try:
            values = [int(cell.strip()) for cell in row]
        except ValueError as exc:
            raise RuntimeError(
                f"{path.name}, red {line_number}: necelobrojna vrednost."
            ) from exc

        require(
            len(set(values)) == K
            and all(1 <= value <= N for value in values)
            and values == sorted(values),
            f"{path.name}, red {line_number}: nelegalna kombinacija.",
        )
        rows.append(values)

    require(len(rows) >= 5, f"{path.name}: nedovoljno kola.")

    return {
        "path": path,
        "data": np.asarray(rows, dtype=np.int64) - 1,
        "digest": hashlib.sha256(raw).hexdigest(),
    }


def rank_many(combinations, n=N, k=K):
    values = np.asarray(combinations, dtype=np.int64)

    require(
        values.ndim == 2
        and values.shape[1] == k
        and 1 <= k <= K
        and k <= n <= N
        and np.all((values >= 0) & (values < n))
        and np.all(np.diff(values, axis=1) > 0),
        "Neispravni skupovi za adresiranje.",
    )

    ranks = np.zeros(len(values), dtype=np.int64)
    start = np.zeros(len(values), dtype=np.int64)

    for position in range(k):
        remaining = k - position
        ranks += (
            BINOMIAL[n - start, remaining]
            - BINOMIAL[n - values[:, position], remaining]
        )
        start = values[:, position] + 1

    require(
        np.all((ranks >= 0) & (ranks < math.comb(n, k))),
        "Adresa kombinacije je van opsega.",
    )

    return ranks


def unrank_combination(address, n=N, k=K):
    address = int(address)
    require(
        0 <= address < math.comb(n, k),
        "Neispravna literalna adresa.",
    )

    residual = address
    start = 0
    result = []
    steps = 0

    for position in range(k):
        remaining = k - position - 1
        found = False

        for value in range(start, n - remaining):
            steps += 1
            require(
                steps <= MAX_PROGRAM_STEPS,
                "Literalni program je prekoracio ogranicenje.",
            )

            block = math.comb(n - value - 1, remaining)

            if residual < block:
                result.append(value)
                start = value + 1
                found = True
                break

            residual -= block

        require(found, "Literalni program nije dekodirao adresu.")

    require(residual == 0, "Preostala je nedekodirana adresa.")
    return np.asarray(result, dtype=np.int64)


def history_inputs(data, times):
    times = np.asarray(times, dtype=np.int64)

    require(
        times.ndim == 1
        and len(times) > 0
        and np.all((times >= 1) & (times <= len(data))),
        "Neispravna istorijska vremena.",
    )

    lag_one = data[times - 1].copy()

    # Kada pretprethodno kolo jos ne postoji, komanda READ_LAG_2
    # eksplicitno cita jedino raspolozivo prethodno kolo.
    lag_two = data[np.maximum(times - 2, 0)].copy()

    patterns = np.zeros((len(times), 2 * N + 1), dtype=bool)
    rows = np.arange(len(times))[:, None]

    patterns[rows, lag_one] = True
    available = times >= 2
    available_rows = np.flatnonzero(available)

    if len(available_rows):
        patterns[
            available_rows[:, None],
            N + lag_two[available_rows],
        ] = True

    patterns[:, -1] = available

    return {
        "patterns": patterns,
        "lag_one": lag_one,
        "lag_two": lag_two,
    }


def slice_history(history, selection):
    return {
        name: values[selection]
        for name, values in history.items()
    }


def fit_transform(patterns, training_rows):
    require(
        1 <= training_rows <= len(patterns),
        "Neispravna granica skaliranja.",
    )

    training = patterns[:training_rows].astype(float)
    mean = training.mean(axis=0)
    scale = training.std(axis=0)

    return {
        "mean": mean,
        "scale": scale,
        "keep": scale > 1e-12,
    }


def transform_features(patterns, transform):
    keep = transform["keep"]
    features = np.ones(
        (len(patterns), 1 + int(np.sum(keep))),
        dtype=np.float64,
    )

    if np.any(keep):
        features[:, 1:] = (
            patterns[:, keep].astype(float)
            - transform["mean"][keep]
        ) / transform["scale"][keep]

    require(np.isfinite(features).all(), "Nekonacne karakteristike.")
    return features


def base_programs():
    programs = []

    for lag in BASE_LAGS:
        for multiplier in BASE_MULTIPLIERS:
            require(
                math.gcd(multiplier, N) == 1,
                "AFFINE mnozilac nije invertibilan modulo 39.",
            )

            for offset in BASE_OFFSETS:
                programs.append({
                    "kind": "affine",
                    "lag": lag,
                    "multiplier": multiplier,
                    "offset": offset,
                })

    return programs


def correction_permutation(source, target):
    """
    Transpozicije daju bijekciju svih 39 brojeva.
    Na izabranom trening slucaju preslikavaju source u target.
    """
    permutation = np.arange(N, dtype=np.int64)
    removed = sorted(set(source.tolist()) - set(target.tolist()))
    added = sorted(set(target.tolist()) - set(source.tolist()))

    require(len(removed) == len(added), "Neispravna programska korekcija.")

    for left, right in zip(removed, added):
        permutation[left] = right
        permutation[right] = left

    require(
        len(np.unique(permutation)) == N
        and np.array_equal(np.sort(permutation[source]), target),
        "Korekcija nije legalna permutacija.",
    )

    return permutation


def resolve_program(program, pattern, lag_one, lag_two):
    steps = 1

    if program["kind"] == "affine":
        source = lag_one if program["lag"] == 1 else lag_two
        permutation = (
            program["multiplier"] * np.arange(N)
            + program["offset"]
        ) % N

    elif program["kind"] == "diagonal":
        node = program["root"]

        while "feature" in node:
            steps += 1
            require(
                steps <= MAX_PROGRAM_STEPS,
                "Program je prekoracio ogranicenje koraka.",
            )
            node = (
                node["one"]
                if pattern[node["feature"]]
                else node["zero"]
            )

        permutation = node["permutation"]
        source = lag_one

    else:
        raise RuntimeError("Nepoznat programski opkod.")

    steps += K

    require(
        steps <= MAX_PROGRAM_STEPS
        and len(np.unique(permutation)) == N,
        "Program nije ogranicena bijektivna transformacija.",
    )

    return permutation, source, steps


def execute_program(program, history):
    if program["kind"] == "affine":
        source = (
            history["lag_one"]
            if program["lag"] == 1
            else history["lag_two"]
        )
        return np.sort(
            (
                program["multiplier"] * source
                + program["offset"]
            ) % N,
            axis=1,
        )

    result = np.empty_like(history["lag_one"])

    for row in range(len(result)):
        permutation, source, _ = resolve_program(
            program,
            history["patterns"][row],
            history["lag_one"][row],
            history["lag_two"][row],
        )
        result[row] = np.sort(permutation[source])

    return result


def tree_signature(node):
    if "feature" not in node:
        return ("leaf", tuple(node["permutation"].tolist()))

    return (
        "if",
        node["feature"],
        tree_signature(node["zero"]),
        tree_signature(node["one"]),
    )


def program_signature(program):
    if program["kind"] == "affine":
        return (
            "affine",
            program["lag"],
            program["multiplier"],
            program["offset"],
        )

    return ("diagonal", tree_signature(program["root"]))


def synthesize_diagonal(programs, history, targets):
    """
    Jedan razlicit istorijski kontekst po obuhvacenom programu.

    Grananje razdvaja ulazne reci bez treniranja klasifikatora.
    Ciljni skupovi odredjuju permutacije u listovima.

    Novi D zadovoljava:
        D(H_i) = stvarni_cilj_i != P_i(H_i).
    """
    target_ranks = rank_many(targets)
    patterns = history["patterns"]
    used_contexts = set()
    witnesses = []
    count = len(targets)

    for program_index, program in enumerate(programs):
        predicted = execute_program(program, history)
        predicted_ranks = rank_many(predicted)

        overlap = np.sum(
            np.any(
                predicted[:, :, None] == targets[:, None, :],
                axis=2,
            ),
            axis=1,
        )
        errors = K - overlap

        # Najveca greska; pri izjednacenju noviji poznati slucaj.
        order = np.lexsort((
            -np.arange(count, dtype=np.int64),
            -errors,
        ))

        selected = None

        for row in order:
            if predicted_ranks[row] == target_ranks[row]:
                continue

            key = patterns[row].tobytes()

            if key not in used_contexts:
                selected = int(row)
                used_contexts.add(key)
                break

        if selected is None:
            continue

        witnesses.append({
            "program": program_index,
            "row": selected,
            "pattern": patterns[selected].copy(),
            "permutation": correction_permutation(
                history["lag_one"][selected],
                targets[selected],
            ),
        })

    if not witnesses:
        return None

    witness_patterns = np.stack([
        witness["pattern"] for witness in witnesses
    ])

    def construct(indices):
        if len(indices) == 1:
            return {
                "permutation": witnesses[int(indices[0])][
                    "permutation"
                ].copy()
            }

        ones = witness_patterns[indices].sum(axis=0)
        valid = (ones > 0) & (ones < len(indices))

        require(
            np.any(valid),
            "Dijagonalni ulazi nisu medjusobno razliciti.",
        )

        balance = np.where(
            valid,
            np.abs(2 * ones - len(indices)),
            np.inf,
        )
        feature = int(np.argmin(balance))
        selected_bits = witness_patterns[indices, feature]

        return {
            "feature": feature,
            "zero": construct(indices[~selected_bits]),
            "one": construct(indices[selected_bits]),
        }

    program = {
        "kind": "diagonal",
        "root": construct(np.arange(len(witnesses))),
        "witnesses": tuple(
            (witness["program"], witness["row"])
            for witness in witnesses
        ),
    }

    result = execute_program(program, history)

    for parent_index, row in program["witnesses"]:
        parent = execute_program(
            programs[parent_index],
            slice_history(history, slice(row, row + 1)),
        )[0]

        require(
            np.array_equal(result[row], targets[row])
            and not np.array_equal(result[row], parent),
            "Dijagonalna konstrukcija ne zadovoljava izabrani slucaj.",
        )

    return program


def build_banks(history, targets, maximum_level):
    current = base_programs()
    banks = {0: tuple(current)}
    signatures = {program_signature(program) for program in current}

    for level in range(1, maximum_level + 1):
        child = synthesize_diagonal(current, history, targets)

        if child is not None:
            signature = program_signature(child)

            if signature not in signatures:
                current.append(child)
                signatures.add(signature)

        require(
            len(current) + 1 <= PROGRAM_DIM,
            "Banka programa ne staje u programski registar.",
        )
        banks[level] = tuple(current)

    return banks


def program_addresses(programs, history):
    return np.stack([
        rank_many(execute_program(program, history))
        for program in programs
    ], axis=1)


def log_amplitudes(amplitudes):
    magnitude = np.abs(amplitudes)
    result = np.full(magnitude.shape, -np.inf)
    positive = magnitude > 0.0
    result[positive] = 2.0 * np.log(magnitude[positive])
    return result


def truncated_statistics(log_bits, limit):
    """
    Nezavisni kvantni bitovi uslovljeni na adresu < limit.
    Bit 0 je najmanje znacajan.

    Vracaju se tacan normalizator i uslovne bitne marginale.
    """
    logs = np.asarray(log_bits, dtype=np.float64)
    require(
        logs.ndim == 3
        and logs.shape[2] == 2
        and not np.isnan(logs).any()
        and not np.isposinf(logs).any(),
        "Neispravne literalne bitne mase.",
    )

    batch, width, _ = logs.shape
    require(
        batch > 0 and 1 <= limit <= (1 << width),
        "Neispravan opseg literalnog programa.",
    )

    upper = int(limit) - 1
    forward = np.full((batch, width + 1, 2), -np.inf)
    backward = np.full_like(forward, -np.inf)

    # Stanje 0: prefiks je manji; stanje 1: jednak gornjoj granici.
    forward[:, 0, 1] = 0.0

    for position in range(width):
        bit = width - position - 1
        upper_bit = (upper >> bit) & 1
        zero = logs[:, bit, 0]
        one = logs[:, bit, 1]
        bit_total = np.logaddexp(zero, one)

        free = forward[:, position, 0] + bit_total
        equal = forward[:, position, 1]

        if upper_bit == 0:
            forward[:, position + 1, 0] = free
            forward[:, position + 1, 1] = equal + zero
        else:
            forward[:, position + 1, 0] = np.logaddexp(
                free, equal + zero
            )
            forward[:, position + 1, 1] = equal + one

    log_z = logsumexp(forward[:, -1], axis=1)

    if not np.isfinite(log_z).all():
        raise DistributionError("Literalni program nema validnu masu.")

    backward[:, -1, :] = 0.0

    for position in range(width - 1, -1, -1):
        bit = width - position - 1
        upper_bit = (upper >> bit) & 1
        zero = logs[:, bit, 0]
        one = logs[:, bit, 1]

        next_free = backward[:, position + 1, 0]
        next_equal = backward[:, position + 1, 1]

        backward[:, position, 0] = np.logaddexp(
            zero + next_free, one + next_free
        )
        backward[:, position, 1] = (
            zero + next_equal
            if upper_bit == 0
            else np.logaddexp(
                zero + next_free, one + next_equal
            )
        )

    marginals = np.zeros_like(logs)

    for position in range(width):
        bit = width - position - 1
        upper_bit = (upper >> bit) & 1

        for value in (0, 1):
            mass = (
                forward[:, position, 0]
                + logs[:, bit, value]
                + backward[:, position + 1, 0]
            )

            if value <= upper_bit:
                destination = 1 if value == upper_bit else 0
                mass = np.logaddexp(
                    mass,
                    forward[:, position, 1]
                    + logs[:, bit, value]
                    + backward[:, position + 1, destination],
                )

            marginals[:, bit, value] = np.exp(mass - log_z)

    require(
        np.all(log_z <= 2e-9)
        and np.allclose(
            marginals.sum(axis=2), 1.0,
            atol=2e-8, rtol=0.0,
        ),
        "Literalna uslovna raspodela nije normalizovana.",
    )

    return {"log_z": log_z, "marginals": marginals}


def truncated_map(log_bits, limit):
    logs = np.asarray(log_bits, dtype=np.float64)
    require(logs.ndim == 2 and logs.shape[1] == 2, "Neispravni bitovi.")

    width = len(logs)
    require(1 <= limit <= (1 << width), "Neispravna granica adrese.")
    upper = int(limit) - 1

    maximum = np.full((width + 1, 2), -np.inf)
    choices = np.zeros((width, 2), dtype=np.int64)
    maximum[-1] = 0.0

    for position in range(width - 1, -1, -1):
        bit = width - position - 1
        upper_bit = (upper >> bit) & 1

        for state in (0, 1):
            best = -np.inf
            selected = 0

            for value in (0, 1):
                if state == 1 and value > upper_bit:
                    continue

                destination = (
                    1
                    if state == 1 and value == upper_bit
                    else 0
                )
                score = (
                    logs[bit, value]
                    + maximum[position + 1, destination]
                )

                if score > best:
                    best = score
                    selected = value

            maximum[position, state] = best
            choices[position, state] = selected

    if not np.isfinite(maximum[0, 1]):
        raise DistributionError("Literalni MAP nema validnu adresu.")

    state = 1
    address = 0

    for position in range(width):
        bit = width - position - 1
        upper_bit = (upper >> bit) & 1
        value = int(choices[position, state])
        address = (address << 1) | value

        if state == 1 and value < upper_bit:
            state = 0

    require(0 <= address < limit, "MAP adresa je van opsega.")
    return address


def initial_parameters(feature_count, restart):
    index = np.arange(
        GATE_COUNT * feature_count, dtype=float
    ).reshape(GATE_COUNT, feature_count)

    theta = (
        0.025 * np.sin(index + 1.0 + 0.71 * restart)
        / np.sqrt(feature_count)
    )
    gates = np.arange(1, GATE_COUNT + 1, dtype=float)
    theta[:, 0] = (
        np.pi / 2.0
        + 0.17 * np.sin(gates)
        + 0.20 * restart * np.cos(gates * np.sqrt(2.0))
    )
    return theta.ravel()


def regularization_mask(feature_count):
    mask = np.ones((GATE_COUNT, feature_count))
    mask[:, 0] = 0.0
    return mask.ravel()


def model_forward(theta, features, active_count, record=False):
    require(
        1 <= active_count <= PROGRAM_DIM,
        "Neispravna velicina programske banke.",
    )

    weights = theta.reshape(GATE_COUNT, features.shape[1])
    angles = features @ weights.T
    amplitudes = np.stack([
        np.cos(angles / 2.0),
        np.sin(angles / 2.0),
    ], axis=2)

    selector_amplitudes = amplitudes[:, :PROGRAM_QUBITS]
    selector_logs = log_amplitudes(selector_amplitudes)

    factors = np.where(
        PROGRAM_RAW_BITS[None, :, :] == 1,
        selector_amplitudes[:, None, :, 1] ** 2,
        selector_amplitudes[:, None, :, 0] ** 2,
    )
    raw_logs = np.sum(np.where(
        PROGRAM_RAW_BITS[None, :, :] == 1,
        selector_logs[:, None, :, 1],
        selector_logs[:, None, :, 0],
    ), axis=2)

    active_raw = PROGRAM_INVERSE[:active_count]
    active_logs = raw_logs[:, active_raw]
    selector_log_z = logsumexp(active_logs, axis=1)

    if not np.isfinite(selector_log_z).all():
        raise DistributionError("Programski registar nema aktivnu masu.")

    log_q = active_logs - selector_log_z[:, None]

    literal_amplitudes = amplitudes[:, PROGRAM_QUBITS:]
    literal_logs = log_amplitudes(literal_amplitudes)
    literal = truncated_statistics(
        literal_logs, TOTAL_COMBINATIONS
    )

    prefix = None
    suffix = None

    if record:
        prefix = np.ones(
            (len(features), PROGRAM_DIM, PROGRAM_QUBITS + 1)
        )
        suffix = np.ones_like(prefix)

        for bit in range(PROGRAM_QUBITS):
            prefix[:, :, bit + 1] = (
                prefix[:, :, bit] * factors[:, :, bit]
            )

        for bit in range(PROGRAM_QUBITS - 1, -1, -1):
            suffix[:, :, bit] = (
                suffix[:, :, bit + 1] * factors[:, :, bit]
            )

    require(
        np.allclose(
            logsumexp(log_q, axis=1), 0.0,
            atol=2e-9, rtol=0.0,
        ),
        "Programske mase nisu normalizovane.",
    )

    return {
        "angles": angles,
        "amplitudes": amplitudes,
        "log_q": log_q,
        "selector_log_z": selector_log_z,
        "active_raw": active_raw,
        "literal_logs": literal_logs,
        "literal": literal,
        "prefix": prefix,
        "suffix": suffix,
    }


def emission_logs(forward, targets, short_addresses):
    bits = (
        targets[:, None]
        >> np.arange(LITERAL_QUBITS)[None, :]
    ) & 1

    literal_log_mass = np.take_along_axis(
        forward["literal_logs"], bits[:, :, None], axis=2
    )[:, :, 0].sum(axis=1) - forward["literal"]["log_z"]

    emissions = np.full_like(forward["log_q"], -np.inf)
    emissions[:, 0] = literal_log_mass
    emissions[:, 1:] = np.where(
        short_addresses == targets[:, None], 0.0, -np.inf
    )

    return emissions, bits


def objective(
    theta, features, targets, short_addresses, weight,
    with_gradient=True,
):
    require(
        len(features) == len(targets) == len(short_addresses)
        and len(targets) > 0,
        "Neispravan trening skup.",
    )

    gradient = np.zeros((GATE_COUNT, features.shape[1]))
    loss_sum = 0.0
    active_count = short_addresses.shape[1] + 1

    for start in range(0, len(targets), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(targets))
        batch_features = features[start:stop]

        try:
            forward = model_forward(
                theta, batch_features, active_count,
                record=with_gradient,
            )
        except DistributionError:
            return np.inf, np.zeros_like(theta)

        emissions, target_bits = emission_logs(
            forward, targets[start:stop], short_addresses[start:stop]
        )
        log_probability = logsumexp(
            forward["log_q"] + emissions, axis=1
        )

        if not np.isfinite(log_probability).all():
            return np.inf, np.zeros_like(theta)

        require(
            np.all(log_probability <= 2e-8),
            "Programska raspodela daje masu vecu od jedan.",
        )
        loss_sum += float(np.maximum(-log_probability, 0.0).sum())

        if not with_gradient:
            continue

        responsibilities = np.exp(
            forward["log_q"] + emissions - log_probability[:, None]
        )

        require(
            np.allclose(
                responsibilities.sum(axis=1), 1.0,
                atol=2e-9, rtol=0.0,
            ),
            "Neispravne programske odgovornosti.",
        )

        with np.errstate(over="ignore", invalid="ignore"):
            active_adjoint = (
                np.exp(-forward["selector_log_z"])[:, None]
                - np.exp(
                    emissions
                    - log_probability[:, None]
                    - forward["selector_log_z"][:, None]
                )
            )

        if not np.isfinite(active_adjoint).all():
            return np.inf, np.zeros_like(theta)

        raw_adjoint = np.zeros((stop - start, PROGRAM_DIM))
        raw_adjoint[:, forward["active_raw"]] = active_adjoint

        angle_gradient = np.zeros((stop - start, GATE_COUNT))

        for bit in range(PROGRAM_QUBITS):
            local_derivative = (
                0.5
                * np.sin(forward["angles"][:, bit])[:, None]
                * (2.0 * PROGRAM_RAW_BITS[None, :, bit] - 1.0)
            )
            derivative = (
                forward["prefix"][:, :, bit]
                * forward["suffix"][:, :, bit + 1]
                * local_derivative
            )
            angle_gradient[:, bit] = np.sum(
                raw_adjoint * derivative, axis=1
            )

        literal_log_gradient = (
            forward["literal"]["marginals"].copy()
        )
        literal_log_gradient[
            np.arange(stop - start)[:, None],
            np.arange(LITERAL_QUBITS)[None, :],
            target_bits,
        ] -= 1.0
        literal_log_gradient *= responsibilities[:, 0, None, None]

        literal_amplitudes = forward["amplitudes"][:, PROGRAM_QUBITS:]
        amplitude_derivatives = np.stack([
            -0.5 * literal_amplitudes[:, :, 1],
            0.5 * literal_amplitudes[:, :, 0],
        ], axis=2)

        contributions = np.zeros_like(literal_amplitudes)

        with np.errstate(
            divide="ignore", invalid="ignore", over="ignore"
        ):
            np.divide(
                2.0 * literal_log_gradient * amplitude_derivatives,
                literal_amplitudes,
                out=contributions,
                where=literal_amplitudes != 0.0,
            )

        if not np.isfinite(contributions).all():
            return np.inf, np.zeros_like(theta)

        angle_gradient[:, PROGRAM_QUBITS:] = contributions.sum(axis=2)
        gradient += angle_gradient.T @ batch_features

    mask = regularization_mask(features.shape[1])
    loss = (
        loss_sum / len(targets)
        + weight * np.dot(mask * theta, theta) / GATE_COUNT
    )
    full_gradient = (
        gradient.ravel() / len(targets)
        + 2.0 * weight * mask * theta / GATE_COUNT
    )

    require(
        np.isfinite(loss) and np.isfinite(full_gradient).all(),
        "Nekonacan gubitak ili gradijent.",
    )
    return float(loss), full_gradient


def fit_model(features, targets, addresses, weight):
    best = None

    for restart in range(RESTARTS):
        tracked = {"loss": np.inf, "theta": None, "calls": 0}

        def tracked_objective(candidate):
            value, gradient = objective(
                candidate, features, targets, addresses, weight
            )
            tracked["calls"] += 1

            if np.isfinite(value) and value < tracked["loss"]:
                tracked["loss"] = value
                tracked["theta"] = candidate.copy()

            if DIAGNOSTICS and tracked["calls"] % 20 == 0:
                detail(
                    f"Pocetak={restart + 1}, "
                    f"procena={tracked['calls']}, "
                    f"NLL+L2={tracked['loss']:.9f}"
                )

            return value, gradient

        result = minimize(
            tracked_objective,
            initial_parameters(features.shape[1], restart),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": MAX_ITERATIONS,
                "gtol": GRADIENT_TOLERANCE,
                "ftol": 1e-10,
                "maxls": 30,
                "maxcor": 10,
            },
        )

        require(
            tracked["theta"] is not None,
            "Optimizator nije nasao konacan kandidat.",
        )

        theta = tracked["theta"]
        loss, gradient = objective(
            theta, features, targets, addresses, weight
        )
        gradient_norm = float(np.max(np.abs(gradient)))

        candidate = {
            "theta": theta.copy(),
            "loss": loss,
            "gradient": gradient_norm,
            "converged": gradient_norm <= GRADIENT_TOLERANCE,
            "message": str(result.message),
        }

        if best is None or loss < best["loss"]:
            best = candidate

    return best


def mixture_map(
    log_q, literal_logs, short_addresses,
    limit=TOTAL_COMBINATIONS, n=N, k=K,
):
    literal = truncated_statistics(
        literal_logs[None, :, :], limit
    )
    literal_mode = truncated_map(literal_logs, limit)

    candidates = np.unique(np.concatenate([
        np.array([literal_mode], dtype=np.int64),
        np.asarray(short_addresses, dtype=np.int64),
    ]))

    bits = (
        candidates[:, None]
        >> np.arange(len(literal_logs))[None, :]
    ) & 1

    literal_scores = literal_logs[
        np.arange(len(literal_logs))[None, :], bits
    ].sum(axis=1) - literal["log_z"][0]

    matches = (
        np.asarray(short_addresses)[:, None]
        == candidates[None, :]
    )
    short_scores = logsumexp(
        np.where(matches, log_q[1:, None], -np.inf),
        axis=0,
    )

    scores = np.logaddexp(
        log_q[0] + literal_scores, short_scores
    )
    index = int(np.argmax(scores))

    require(
        np.isfinite(scores[index]) and scores[index] <= 2e-8,
        "Neispravan globalni programski MAP.",
    )

    address = int(candidates[index])

    return {
        "address": address,
        "combination": unrank_combination(address, n=n, k=k),
        "log_probability": float(scores[index]),
        "literal_marginals": literal["marginals"][0],
    }


def evaluate(theta, features, targets, addresses):
    losses = []
    hits = []

    for start in range(0, len(targets), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(targets))
        forward = model_forward(
            theta, features[start:stop], addresses.shape[1] + 1
        )
        emissions, _ = emission_logs(
            forward, targets[start:stop], addresses[start:stop]
        )
        log_probability = logsumexp(
            forward["log_q"] + emissions, axis=1
        )

        require(
            not np.isnan(log_probability).any()
            and np.all(log_probability <= 2e-8),
            "Neispravna evaluaciona masa.",
        )
        losses.extend(
            np.maximum(-log_probability, 0.0).tolist()
        )

        for row in range(stop - start):
            result = mixture_map(
                forward["log_q"][row],
                forward["literal_logs"][row],
                addresses[start + row],
            )
            target = unrank_combination(targets[start + row])
            hits.append(
                len(
                    set(result["combination"].tolist())
                    & set(target.tolist())
                )
            )

    require(len(losses) > 0, "Prazna evaluacija.")
    losses = np.asarray(losses)

    return {
        "nll": float(losses.mean()),
        "hits": np.asarray(hits, dtype=np.int64),
        "zero_targets": int(np.isposinf(losses).sum()),
    }


def qiskit_program_output(program, history):
    require(len(history["patterns"]) == 1, "Potreban je jedan kontekst.")

    permutation, source, _ = resolve_program(
        program,
        history["patterns"][0],
        history["lag_one"][0],
        history["lag_two"][0],
    )

    full_permutation = np.arange(OUTPUT_DIM)
    full_permutation[:N] = permutation
    unitary = np.zeros((OUTPUT_DIM, OUTPUT_DIM), dtype=np.complex128)
    unitary[full_permutation, np.arange(OUTPUT_DIM)] = 1.0
    gate = UnitaryGate(unitary, label="Program_permutation")

    actual_numbers = []

    for value in source:
        circuit = QuantumCircuit(OUTPUT_QUBITS)

        for bit in range(OUTPUT_QUBITS):
            if (int(value) >> bit) & 1:
                circuit.x(bit)

        circuit.append(gate, range(OUTPUT_QUBITS))
        actual = Statevector.from_instruction(circuit).data

        expected = np.zeros(OUTPUT_DIM, dtype=np.complex128)
        expected[permutation[value]] = 1.0

        require(
            np.allclose(
                actual, expected,
                atol=2e-10, rtol=0.0,
            ),
            "Qiskit programska grana ne izvodi permutaciju.",
        )
        actual_numbers.append(int(np.argmax(np.abs(actual) ** 2)))

    combination = np.sort(np.asarray(actual_numbers))

    require(
        len(np.unique(combination)) == K,
        "Programska grana nije sacuvala razlicite brojeve.",
    )
    return combination


def qiskit_forward(theta, feature, programs, history):
    weights = theta.reshape(GATE_COUNT, len(feature))
    angles = weights @ feature

    circuit = QuantumCircuit(PROGRAM_QUBITS)

    for bit in range(PROGRAM_QUBITS):
        circuit.ry(float(angles[bit]), bit)

    for control in range(PROGRAM_QUBITS - 1):
        circuit.cx(control, control + 1)

    selector_state = np.asarray(
        Statevector.from_instruction(circuit).data
    )

    raw_amplitudes = np.prod(np.where(
        PROGRAM_RAW_BITS == 1,
        np.sin(angles[:PROGRAM_QUBITS] / 2.0)[None, :],
        np.cos(angles[:PROGRAM_QUBITS] / 2.0)[None, :],
    ), axis=1)

    require(
        np.allclose(
            selector_state,
            raw_amplitudes[PROGRAM_INVERSE],
            atol=2e-10, rtol=2e-9,
        ),
        "Qiskit programske amplitude nisu dosledne.",
    )

    active_count = len(programs) + 1
    log_q = log_amplitudes(selector_state[:active_count])
    selector_log_z = float(logsumexp(log_q))

    if not np.isfinite(selector_log_z):
        raise DistributionError("Qiskit nema aktivnu programsku masu.")

    log_q -= selector_log_z

    literal_logs = np.empty((LITERAL_QUBITS, 2))

    for bit in range(LITERAL_QUBITS):
        literal_circuit = QuantumCircuit(1)
        literal_circuit.ry(
            float(angles[PROGRAM_QUBITS + bit]), 0
        )
        actual = Statevector.from_instruction(literal_circuit).data
        literal_logs[bit] = log_amplitudes(actual)

    literal = truncated_statistics(
        literal_logs[None, :, :], TOTAL_COMBINATIONS
    )

    addresses = np.array([
        rank_many(
            qiskit_program_output(program, history)[None, :]
        )[0]
        for program in programs
    ], dtype=np.int64)

    numerical = model_forward(
        theta, feature[None, :], active_count
    )
    expected_addresses = program_addresses(programs, history)[0]

    require(
        np.array_equal(addresses, expected_addresses)
        and np.allclose(
            np.exp(log_q),
            np.exp(numerical["log_q"][0]),
            atol=2e-9, rtol=2e-8,
        )
        and np.allclose(
            np.exp(literal_logs),
            np.exp(numerical["literal_logs"][0]),
            atol=2e-10, rtol=2e-9,
        )
        and np.allclose(
            literal["marginals"][0],
            numerical["literal"]["marginals"][0],
            atol=2e-9, rtol=2e-8,
        ),
        "Qiskit i numericka programska raspodela nisu dosledni.",
    )

    result = mixture_map(log_q, literal_logs, addresses)
    result["literal_mass"] = float(np.exp(log_q[0]))
    return result


def self_checks():
    require(
        len(np.unique(PROGRAM_PERMUTATION)) == PROGRAM_DIM,
        "Programski CNOT nije permutacija.",
    )

    toy_combinations = np.array(
        list(itertools.combinations(range(5), 3)),
        dtype=np.int64,
    )
    toy_addresses = rank_many(toy_combinations, n=5, k=3)

    require(
        np.array_equal(
            toy_addresses, np.arange(len(toy_combinations))
        ),
        "Kanonsko adresiranje nije tacno.",
    )

    for address, combination in enumerate(toy_combinations):
        require(
            np.array_equal(
                unrank_combination(address, n=5, k=3),
                combination,
            ),
            "Literalni dekoder nije tacan.",
        )

    for address in (0, TOTAL_COMBINATIONS // 2, TOTAL_COMBINATIONS - 1):
        combination = unrank_combination(address)
        require(
            rank_many(combination[None, :])[0] == address,
            "Stvarna literalna adresa nije povratno dekodirana.",
        )

    angles = np.array([0.7, 1.1, 1.8, 2.1, 0.9])
    amplitudes = np.stack([
        np.cos(angles / 2.0),
        np.sin(angles / 2.0),
    ], axis=1)
    logs = log_amplitudes(amplitudes)
    limit = 23

    statistics = truncated_statistics(logs[None, :, :], limit)
    codes = np.arange(limit)
    bits = (
        codes[:, None] >> np.arange(len(angles))[None, :]
    ) & 1

    scores = logs[np.arange(len(angles))[None, :], bits].sum(axis=1)
    masses = np.exp(scores - logsumexp(scores))
    brute_marginals = np.zeros_like(logs)

    for row, mass in zip(bits, masses):
        brute_marginals[np.arange(len(angles)), row] += mass

    require(
        np.isclose(
            statistics["log_z"][0],
            logsumexp(scores),
            atol=1e-12,
        )
        and np.allclose(
            statistics["marginals"][0],
            brute_marginals,
            atol=1e-12, rtol=0.0,
        )
        and truncated_map(logs, limit) == int(np.argmax(scores)),
        "Literalna normalizacija, marginale ili MAP nisu tacni.",
    )

    # Potpuna provera mesavine, ukljucujuci iste izlaze dva programa.
    small_angles = angles[:4]
    small_logs = log_amplitudes(np.stack([
        np.cos(small_angles / 2.0),
        np.sin(small_angles / 2.0),
    ], axis=1))
    log_q = np.log(np.array([0.4, 0.3, 0.3]))
    short = np.array([1, 1], dtype=np.int64)

    prediction = mixture_map(
        log_q, small_logs, short,
        limit=10, n=5, k=3,
    )

    all_codes = np.arange(10)
    all_bits = (
        all_codes[:, None] >> np.arange(4)[None, :]
    ) & 1
    literal_scores = small_logs[
        np.arange(4)[None, :], all_bits
    ].sum(axis=1)
    literal_masses = np.exp(
        literal_scores - logsumexp(literal_scores)
    )
    mixture = 0.4 * literal_masses
    mixture[1] += 0.6

    require(
        np.isclose(mixture.sum(), 1.0, atol=1e-12)
        and prediction["address"] == int(np.argmax(mixture))
        and np.isclose(
            prediction["log_probability"],
            np.log(mixture.max()),
            atol=1e-12,
        ),
        "Programske mase ili globalni MAP nisu tacni.",
    )

    fixture = np.array([
        [0, 4, 9, 13, 20, 30, 38],
        [1, 6, 10, 14, 22, 31, 37],
        [0, 5, 12, 16, 24, 33, 38],
        [2, 7, 11, 18, 23, 29, 36],
        [1, 8, 14, 19, 26, 32, 37],
        [3, 9, 15, 21, 27, 34, 38],
        [0, 6, 13, 18, 25, 31, 36],
        [2, 8, 12, 19, 24, 32, 38],
    ], dtype=np.int64)

    times = np.arange(1, len(fixture))
    history = history_inputs(fixture, times)
    targets = fixture[1:]
    target_addresses = rank_many(targets)
    cutoff = 5

    banks = build_banks(
        slice_history(history, slice(0, cutoff)),
        targets[:cutoff],
        max(DIAGONAL_LEVELS),
    )
    programs = banks[max(DIAGONAL_LEVELS)]

    require(
        any(program["kind"] == "diagonal" for program in programs),
        "Na provernim podacima nije napravljena dijagonalna dopuna.",
    )

    addresses = program_addresses(programs, history)
    transform = fit_transform(history["patterns"], cutoff)
    features = transform_features(history["patterns"], transform)
    theta = initial_parameters(features.shape[1], 0)

    value, analytic = objective(
        theta, features[:cutoff], target_addresses[:cutoff],
        addresses[:cutoff], REGULARIZATIONS[0],
    )
    require(np.isfinite(value), "Nekonacan proverni gubitak.")

    selected = set(
        np.linspace(0, len(theta) - 1, 18, dtype=int).tolist()
    )

    for gate in (0, PROGRAM_QUBITS - 1, PROGRAM_QUBITS, GATE_COUNT - 1):
        selected.add(gate * features.shape[1])

    step = 1e-6

    for index in sorted(selected):
        upper = theta.copy()
        lower = theta.copy()
        upper[index] += step
        lower[index] -= step

        numeric = (
            objective(
                upper, features[:cutoff],
                target_addresses[:cutoff], addresses[:cutoff],
                REGULARIZATIONS[0], with_gradient=False,
            )[0]
            - objective(
                lower, features[:cutoff],
                target_addresses[:cutoff], addresses[:cutoff],
                REGULARIZATIONS[0], with_gradient=False,
            )[0]
        ) / (2.0 * step)

        require(
            np.isclose(
                analytic[index], numeric,
                atol=3e-5, rtol=4e-4,
            ),
            "Koordinatna provera gradijenta nije prosla.",
        )

    for direction_index in range(3):
        direction = np.sin(
            (np.arange(len(theta)) + 1.0)
            * (0.31 + 0.17 * direction_index)
        )
        direction /= np.linalg.norm(direction)

        numeric = (
            objective(
                theta + step * direction,
                features[:cutoff], target_addresses[:cutoff],
                addresses[:cutoff], REGULARIZATIONS[0],
                with_gradient=False,
            )[0]
            - objective(
                theta - step * direction,
                features[:cutoff], target_addresses[:cutoff],
                addresses[:cutoff], REGULARIZATIONS[0],
                with_gradient=False,
            )[0]
        ) / (2.0 * step)

        require(
            np.isclose(
                analytic @ direction, numeric,
                atol=3e-5, rtol=4e-4,
            ),
            "Usmerena provera gradijenta nije prosla.",
        )

    qiskit_forward(
        theta,
        features[cutoff],
        programs,
        slice_history(history, slice(cutoff, cutoff + 1)),
    )

    changed = fixture.copy()
    changed[cutoff + 1:] = changed[cutoff + 1:][::-1]
    changed_history = history_inputs(changed, times)
    changed_banks = build_banks(
        slice_history(changed_history, slice(0, cutoff)),
        changed[1:cutoff + 1],
        max(DIAGONAL_LEVELS),
    )
    changed_transform = fit_transform(
        changed_history["patterns"], cutoff
    )

    require(
        np.array_equal(
            history["patterns"][:cutoff + 1],
            changed_history["patterns"][:cutoff + 1],
        )
        and tuple(program_signature(p) for p in programs)
        == tuple(
            program_signature(p)
            for p in changed_banks[max(DIAGONAL_LEVELS)]
        )
        and np.array_equal(
            transform["keep"], changed_transform["keep"]
        )
        and np.allclose(
            transform["mean"], changed_transform["mean"]
        )
        and np.allclose(
            transform["scale"], changed_transform["scale"]
        ),
        "Buduci podaci uticu na raniju programsku banku ili skaliranje.",
    )

    detail("Ugradjene programske, dijagonalne, Qiskit, MAP i gradijent provere su prosle.")


def longest_consecutive(values):
    longest = 1
    current = 1

    for difference in np.diff(values):
        current = current + 1 if difference == 1 else 1
        longest = max(longest, current)

    return longest


def process_dataset(dataset):
    data = dataset["data"]
    name = dataset["path"].name

    times = np.arange(1, len(data))
    history = history_inputs(data, times)
    targets = data[1:]
    target_addresses = rank_many(targets)
    total = len(targets)

    train_end = int(TRAIN_FRACTION * total)
    validation_end = int(VALIDATION_END_FRACTION * total)

    require(
        0 < train_end < validation_end < total,
        f"{name}: nedovoljno podataka za hronolosku podelu.",
    )

    print(f"{name} | ucenje programa", flush=True)
    detail(f"Redova={len(data)}, SHA256={dataset['digest']}")

    banks = build_banks(
        slice_history(history, slice(0, train_end)),
        targets[:train_end],
        max(DIAGONAL_LEVELS),
    )
    transform = fit_transform(history["patterns"], train_end)
    features = transform_features(history["patterns"], transform)

    best = None

    for level in DIAGONAL_LEVELS:
        programs = banks[level]
        addresses = program_addresses(programs, history)

        for weight in REGULARIZATIONS:
            fit = fit_model(
                features[:train_end],
                target_addresses[:train_end],
                addresses[:train_end],
                weight,
            )

            try:
                metrics = evaluate(
                    fit["theta"],
                    features[train_end:validation_end],
                    target_addresses[train_end:validation_end],
                    addresses[train_end:validation_end],
                )
            except DistributionError as exc:
                detail(f"Kandidat odbijen: {exc}")
                continue

            detail(
                f"Nivo={level}, L2={weight}, "
                f"validacioni NLL={metrics['nll']:.9f}"
            )

            if np.isfinite(metrics["nll"]) and (
                best is None or metrics["nll"] < best["metrics"]["nll"]
            ):
                best = {
                    "level": level,
                    "weight": weight,
                    "fit": fit,
                    "metrics": metrics,
                    "program_count": len(programs),
                }

    require(best is not None, f"{name}: nema validnog modela.")

    print(
        f"Validacija | programa={best['program_count']}, "
        f"nivo={best['level']}, L2={best['weight']}, "
        f"NLL={best['metrics']['nll']:.6f}, "
        f"pogodaka={best['metrics']['hits'].mean():.4f}",
        flush=True,
    )

    test_programs = build_banks(
        slice_history(history, slice(0, validation_end)),
        targets[:validation_end],
        best["level"],
    )[best["level"]]
    test_addresses = program_addresses(test_programs, history)

    test_transform = fit_transform(
        history["patterns"], validation_end
    )
    test_features = transform_features(
        history["patterns"], test_transform
    )

    test_fit = fit_model(
        test_features[:validation_end],
        target_addresses[:validation_end],
        test_addresses[:validation_end],
        best["weight"],
    )
    test_metrics = evaluate(
        test_fit["theta"],
        test_features[validation_end:],
        target_addresses[validation_end:],
        test_addresses[validation_end:],
    )

    print(
        f"Test | NLL={test_metrics['nll']:.6f}, "
        f"pogodaka={test_metrics['hits'].mean():.4f}",
        flush=True,
    )

    if test_metrics["zero_targets"]:
        print(
            f"Upozorenje: test ciljevi sa nultom masom="
            f"{test_metrics['zero_targets']}"
        )

    # Zavrsna banka i amplitude uce se na svim istorijskim prelazima.
    final_programs = build_banks(
        history, targets, best["level"]
    )[best["level"]]
    final_addresses = program_addresses(final_programs, history)

    final_transform = fit_transform(history["patterns"], total)
    final_features = transform_features(
        history["patterns"], final_transform
    )
    final_fit = fit_model(
        final_features,
        target_addresses,
        final_addresses,
        best["weight"],
    )

    next_history = history_inputs(
        data, np.array([len(data)])
    )
    next_features = transform_features(
        next_history["patterns"], final_transform
    )

    result = qiskit_forward(
        final_fit["theta"],
        next_features[0],
        final_programs,
        next_history,
    )
    combination = result["combination"] + 1

    require(
        len(combination) == K
        and len(set(combination.tolist())) == K
        and np.all((combination >= 1) & (combination <= N)),
        "Nelegalna NEXT kombinacija.",
    )

    unconfirmed = [
        label
        for label, fit in (
            ("validacioni", best["fit"]),
            ("test", test_fit),
            ("zavrsni", final_fit),
        )
        if not fit["converged"]
    ]

    if unconfirmed:
        print(
            "Napomena: mali gradijent nije potvrdjen za "
            + ", ".join(unconfirmed)
            + " model."
        )

    diagonal_count = sum(
        program["kind"] == "diagonal"
        for program in final_programs
    )
    if diagonal_count == 0:
        print(
            "Napomena: nije nadjen raspoloziv skup dijagonalnih "
            "kontraprimera za novu dopunu."
        )

    if np.array_equal(combination, data[-1] + 1):
        print("Upozorenje: NEXT ponavlja poslednje kolo.")

    run = longest_consecutive(combination)
    if run >= 5:
        print(f"Upozorenje: NEXT ima {run} uzastopnih brojeva.")

    span = int(combination[-1] - combination[0])
    span_percentile = float(
        np.mean((data[:, -1] - data[:, 0]) <= span)
    )
    if span_percentile <= 0.01:
        print("Upozorenje: izrazito grupisana NEXT kombinacija.")

    detail(
        f"Dijagonalnih dopuna={diagonal_count}, "
        f"literalna masa={result['literal_mass']:.9f}, "
        f"log P(NEXT)={result['log_probability']:.9f}"
    )

    print(
        f"{name} NEXT: "
        + " ".join(str(int(value)) for value in combination),
        flush=True,
    )

    return {
        "name": name,
        "digest": dataset["digest"],
        "next": combination,
        "literal_mass": result["literal_mass"],
        "literal_marginals": result["literal_marginals"],
        "log_probability": result["log_probability"],
    }


def main():
    print(
        "Quantum Cantor–Turing V1 | "
        "ograniceni programi i konstruktivne dijagonalne dopune"
    )
    print(
        f"Qiskit {qiskit.__version__} | "
        f"{PROGRAM_QUBITS} programskih + "
        f"{LITERAL_QUBITS} literalna pomocna kubita; "
        "7 izlaznih registara po 6 kubita, blokovska simulacija"
    )

    self_checks()

    results = [
        process_dataset(load_csv(path))
        for path in FILES
    ]

    for first, second in itertools.combinations(results, 2):
        if first["digest"] == second["digest"]:
            print(
                f"Upozorenje: isti sadrzaj "
                f"{first['name']} i {second['name']}."
            )

        if np.array_equal(first["next"], second["next"]):
            bit_difference = float(np.max(np.abs(
                first["literal_marginals"]
                - second["literal_marginals"]
            )))

            print(
                f"Isti NEXT: {first['name']} / {second['name']} | "
                f"razlika literalne mase="
                f"{abs(first['literal_mass'] - second['literal_mass']):.3e}, "
                f"bitnih marginala={bit_difference:.3e}, "
                f"log mase NEXT="
                f"{abs(first['log_probability'] - second['log_probability']):.3e}"
            )


if __name__ == "__main__":
    main()



"""
Quantum Cantor-Turing V1 | ograniceni programi i konstruktivne dijagonalne dopune
Qiskit 1.4.4 | 6 programskih + 24 literalna pomocna kubita; 7 izlaznih registara po 6 kubita, blokovska simulacija

loto7_4716_k89.csv | ucenje programa
Validacija | programa=27, nivo=3, L2=0.1, NLL=16.892624, pogodaka=1.2560
Test | NLL=16.814856, pogodaka=1.2726
Napomena: mali gradijent nije potvrdjen za validacioni, test, zavrsni model.
loto7_4716_k89.csv NEXT: 7 x 18 y 30 z 35

loto7_4716_k89_loto_2980.csv | ucenje programa
Validacija | programa=25, nivo=1, L2=0.1, NLL=16.960685, pogodaka=1.2394
Test | NLL=16.985979, pogodaka=1.1834
Napomena: mali gradijent nije potvrdjen za validacioni, test, zavrsni model.
loto7_4716_k89_loto_2980.csv NEXT: 1 x 13 y 16 z 39

loto7_4716_k89_loto_plus_1736.csv | ucenje programa
Validacija | programa=25, nivo=1, L2=0.1, NLL=17.683246, pogodaka=1.2385
Test | NLL=17.286699, pogodaka=1.3257
Napomena: mali gradijent nije potvrdjen za validacioni, test, zavrsni model.
loto7_4716_k89_loto_plus_1736.csv NEXT: 5 x 10 y 16 z 36
"""



"""
Dijagonalni postupak konstruiše objekat koji se razlikuje od svakog objekta na zadatoj listi, koristeći za svaki red odgovarajuće mesto na dijagonali. 
Turing taj postupak primenjuje na programe i njihove izlaze. 

Za Loto model uči programe i konstruktivno proširuje njihov skup.
1. Programi predstavljaju hipoteze o prelazima između kola. Svaki sadrži kratko pravilo koje obrađuje istoriju i formira predikciju celog skupa od sedam brojeva.
2. Istorija određuje težine tih programa. Iz njihovih izlaza učim prediktivnu raspodelu. Programska indukcija ima stvarnu teorijsku vezu sa Turingovim mašinama, razvijenu u Solomonoffovom radu. 
3. Dijagonalna konstrukcija vodi proširivanje hipoteza. Za postojeće programe tražim konkretne istorijske slučajeve u kojima njihove predikcije greše. Ti slučajevi služe za konstrukciju novih pravila koja ispravljaju utvrđene propuste. Razlikovanje programa samo po sebi ne određuje NEXT.
4. Qiskit predstavlja programsku raspodelu amplitudama. Programski registar upravlja ograničenim izvršavanjem pravila, a izlazni registri daju raspodelu kombinacija. Dužina programa i broj koraka unapred su ograničeni.
Pretraga i proširivanje prostora programa. To konkretno određuje učenje i nastanak predikcije.

Koristi ograničene programe, konstruktivne dijagonalne dopune i Qiskit amplitudnu raspodelu nad programima. 
Literalni program obezbeđuje adresiranje svih legalnih kombinacija. 
NEXT se dobija tačnim maksimumom zajedničke programske raspodele.
"""
