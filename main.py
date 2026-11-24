import json
import re
import time
from pathlib import Path

# 부동소수점 오차 보정을 위한 동점 판정 허용오차 (epsilon)
EPSILON = 1e-9
REPEAT_COUNT = 10


class PatternMatrix:
    """n×n 크기의 2차원 패턴 및 필터 데이터를 저장하고 조작하는 클래스"""

    def __init__(self, size: int, grid=None):
        if size <= 0:
            raise ValueError("배열 크기는 1 이상이어야 합니다.")
        self.size = size
        if grid is None:
            self.grid = [[0.0 for _ in range(size)] for _ in range(size)]
        else:
            if len(grid) != size or any(len(row) != size for row in grid):
                raise ValueError(f"크기 불일치: {size}×{size} 정사각 배열이 아닙니다.")
            self.grid = [[float(val) for val in row] for row in grid]

    def get_value(self, row: int, col: int) -> float:
        """특정 위치(row, col)의 값을 읽어옵니다."""
        self._check_bounds(row, col)
        return self.grid[row][col]

    def set_value(self, row: int, col: int, value: float) -> None:
        """특정 위치(row, col)에 값을 저장합니다."""
        self._check_bounds(row, col)
        self.grid[row][col] = float(value)

    def _check_bounds(self, row: int, col: int) -> None:
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise IndexError(f"인덱스 범위 초과: ({row}, {col})는 {self.size}×{self.size} 범위를 벗어났습니다.")

    def to_1d(self) -> list:
        """[보너스] 2차원 배열을 길이 N²의 1차원 배열로 변환합니다."""
        flat = []
        for row in self.grid:
            flat.extend(row)
        return flat


def normalize_label(raw_label: str) -> str:
    """제각각인 라벨 표기('+', 'cross', 'x' 등)를 표준 라벨('Cross', 'X')로 통일합니다."""
    if not isinstance(raw_label, str):
        raise ValueError(f"유효하지 않은 라벨 형식: {raw_label}")
    cleaned = raw_label.strip().lower()
    if cleaned in ("+", "cross"):
        return "Cross"
    if cleaned == "x":
        return "X"
    raise ValueError(f"지원하지 않는 라벨 값입니다: '{raw_label}'")


def compute_mac(pattern: PatternMatrix, filter_matrix: PatternMatrix) -> float:
    """외부 라이브러리 없이 이중 반복문으로 위치별 곱셈 후 누적 합(MAC)을 계산합니다."""
    if pattern.size != filter_matrix.size:
        raise ValueError(
            f"패턴 크기({pattern.size}×{pattern.size})와 필터 크기({filter_matrix.size}×{filter_matrix.size})가 일치하지 않습니다."
        )
    n = pattern.size
    p_grid = pattern.grid
    f_grid = filter_matrix.grid
    total = 0.0
    for r in range(n):
        for c in range(n):
            total += p_grid[r][c] * f_grid[r][c]
    return float(total)


def compute_mac_1d(pattern_1d: list, filter_1d: list) -> float:
    """[보너스] 1차원 배열로 변환된 데이터의 MAC 연산을 수행합니다."""
    if len(pattern_1d) != len(filter_1d):
        raise ValueError("1차원 배열 길이가 일치하지 않습니다.")
    total = 0.0
    for idx in range(len(pattern_1d)):
        total += pattern_1d[idx] * filter_1d[idx]
    return float(total)


def judge_scores(score_a: float, score_b: float, label_a: str, label_b: str, tie_label: str = "UNDECIDED") -> str:
    """허용오차(epsilon = 1e-9) 기준으로 두 점수를 비교해 최종 라벨을 판정합니다."""
    if abs(score_a - score_b) < EPSILON:
        return tie_label
    return label_a if score_a > score_b else label_b


def measure_mac_avg_ms(pattern: PatternMatrix, filter_matrix: PatternMatrix, repeats: int = REPEAT_COUNT) -> float:
    """I/O 시간을 제외하고 순수 MAC 연산 함수 호출 시간만 반복 측정해 평균(ms)을 반환합니다."""
    start = time.perf_counter()
    for _ in range(repeats):
        compute_mac(pattern, filter_matrix)
    elapsed_sec = time.perf_counter() - start
    return (elapsed_sec / repeats) * 1000.0


def generate_synthetic_patterns(n: int) -> tuple:
    """[보너스] 크기 N에 맞는 N×N 십자가(Cross)와 X 패턴을 자동 생성합니다."""
    cross = PatternMatrix(n)
    x_pat = PatternMatrix(n)
    mid = n // 2
    for r in range(n):
        for c in range(n):
            if r == mid or c == mid:
                cross.set_value(r, c, 1.0)
            if r == c or (r + c) == (n - 1):
                x_pat.set_value(r, c, 1.0)
    return cross, x_pat


def read_3x3_matrix(title: str) -> PatternMatrix:
    """콘솔에서 3×3 숫자 배열을 한 줄씩 입력받고 행/열 및 숫자 파싱을 검증합니다."""
    print(f"{title} (3줄 입력, 공백 구분)")
    rows = []
    row_idx = 0
    while row_idx < 3:
        raw_line = input().strip()
        parts = raw_line.split()
        if len(parts) != 3:
            print("입력 형식 오류: 각 줄에 3개의 숫자를 공백으로 구분해 입력하세요.")
            continue
        try:
            nums = [float(x) for x in parts]
        except ValueError:
            print("입력 형식 오류: 각 줄에 3개의 숫자를 공백으로 구분해 입력하세요.")
            continue
        rows.append(nums)
        row_idx += 1
    return PatternMatrix(3, rows)


def print_performance_table(sizes: list, sample_pairs: dict, repeats: int = REPEAT_COUNT) -> None:
    """크기별 평균 연산 시간(ms)과 연산 횟수(N²) 표를 출력합니다."""
    print(f"\n#---------------------------------------")
    print(f"# [성능 분석] (평균/{repeats}회)")
    print(f"#---------------------------------------")
    print(f"{'크기':<10} {'평균 시간(ms)':<16} {'연산 횟수':<10}")
    print("-" * 38)
    for n in sizes:
        pat, flt = sample_pairs[n]
        avg_ms = measure_mac_avg_ms(pat, flt, repeats=repeats)
        ops = n * n
        size_label = f"{n}×{n}"
        print(f"{size_label:<10} {avg_ms:<16.4f} {ops:<10}")


def run_user_input_mode() -> None:
    """모드 1: 사용자 직접 입력 (3×3 필터 A, B 및 패턴 판정)"""
    print("\n#---------------------------------------")
    print("# [1] 필터 입력")
    print("#---------------------------------------")
    filter_a = read_3x3_matrix("필터 A")
    print()
    filter_b = read_3x3_matrix("필터 B")
    print("\n✓ 필터 A, 필터 B (3×3) 저장 확인 완료")

    print("\n#---------------------------------------")
    print("# [2] 패턴 입력")
    print("#---------------------------------------")
    pattern = read_3x3_matrix("패턴")

    score_a = compute_mac(pattern, filter_a)
    score_b = compute_mac(pattern, filter_b)
    avg_ms = measure_mac_avg_ms(pattern, filter_a, repeats=REPEAT_COUNT)
    verdict = judge_scores(score_a, score_b, "A", "B", tie_label="판정 불가")

    print("\n#---------------------------------------")
    print("# [3] MAC 결과")
    print("#---------------------------------------")
    if verdict == "판정 불가":
        print(f"A 점수: {score_a:.16f}")
        print(f"B 점수: {score_b:.16f}")
        print(f"연산 시간(평균/{REPEAT_COUNT}회): {avg_ms:.4f} ms")
        print(f"판정: 판정 불가 (|A-B| < {EPSILON})")
    else:
        print(f"A 점수: {score_a:.1f}")
        print(f"B 점수: {score_b:.1f}")
        print(f"연산 시간(평균/{REPEAT_COUNT}회): {avg_ms:.4f} ms")
        print(f"판정: {verdict}")

    print_performance_table([3], {3: (pattern, filter_a)}, repeats=REPEAT_COUNT)


def run_json_analysis_mode(json_path: str = "data.json") -> None:
    """모드 2: data.json 로드, 스키마 검증, 라벨 정규화, 일괄 판정 및 성능 분석"""
    path = Path(json_path)
    if not path.exists():
        print(f"\n[오류] '{json_path}' 파일을 찾을 수 없습니다. 같은 폴더에 파일을 배치해 주세요.")
        return

    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"\n[오류] JSON 파싱 실패: {exc}")
        return

    raw_filters = raw_data.get("filters")
    raw_patterns = raw_data.get("patterns")
    if not isinstance(raw_filters, dict) or not isinstance(raw_patterns, dict):
        print("\n[오류] data.json 최상위 스키마('filters', 'patterns')가 올바르지 않습니다.")
        return

    print("\n#---------------------------------------")
    print("# [1] 필터 로드")
    print("#---------------------------------------")
    loaded_filters = {}
    for size_key, filter_group in raw_filters.items():
        match = re.match(r"^size_(\d+)$", size_key)
        if not match or not isinstance(filter_group, dict):
            print(f"✗ {size_key} 필터 스키마 오류로 건너뜀")
            continue
        n = int(match.group(1))
        normalized_group = {}
        try:
            for raw_k, matrix_data in filter_group.items():
                std_label = normalize_label(raw_k)
                normalized_group[std_label] = PatternMatrix(n, matrix_data)
            if "Cross" not in normalized_group or "X" not in normalized_group:
                raise ValueError("필수 필터(Cross, X)가 누락되었습니다.")
            loaded_filters[n] = normalized_group
            print(f"✓ {size_key:<7} 필터 로드 완료 (Cross, X)")
        except Exception as exc:
            print(f"✗ {size_key:<7} 필터 로드 실패: {exc}")

    print("\n#---------------------------------------")
    print("# [2] 패턴 분석 (라벨 정규화 적용)")
    print("#---------------------------------------")
    total_count = 0
    pass_count = 0
    fail_cases = []
    perf_samples = {}

    for case_key, case_info in raw_patterns.items():
        total_count += 1
        print(f"-- {case_key} ---")
        try:
            if not isinstance(case_info, dict):
                raise ValueError("케이스 데이터 형식이 객체(dict)가 아닙니다.")

            key_match = re.match(r"^size_(\d+)_(.+)$", case_key)
            if not key_match:
                raise ValueError(f"패턴 키 규칙(size_{{N}}_{{idx}}) 위반: '{case_key}'")

            n = int(key_match.group(1))
            if n not in loaded_filters:
                raise ValueError(f"크기 {n}×{n}에 대응하는 필터(size_{n})가 로드되지 않았습니다.")

            raw_input = case_info.get("input")
            raw_expected = case_info.get("expected")
            if not isinstance(raw_input, list):
                raise ValueError("'input' 필드가 2차원 배열이 아닙니다.")

            pattern_matrix = PatternMatrix(n, raw_input)
            expected_label = normalize_label(raw_expected)

            cross_filter = loaded_filters[n]["Cross"]
            x_filter = loaded_filters[n]["X"]

            if n not in perf_samples:
                perf_samples[n] = (pattern_matrix, cross_filter)

            cross_score = compute_mac(pattern_matrix, cross_filter)
            x_score = compute_mac(pattern_matrix, x_filter)
            verdict = judge_scores(cross_score, x_score, "Cross", "X", tie_label="UNDECIDED")

            print(f"Cross 점수: {cross_score:.1f}" if cross_score.is_integer() else f"Cross 점수: {cross_score:.16f}")
            print(f"X 점수: {x_score:.1f}" if x_score.is_integer() else f"X 점수: {x_score:.16f}")

            if verdict == expected_label:
                pass_count += 1
                print(f"판정: {verdict} | expected: {expected_label} | PASS")
            else:
                if verdict == "UNDECIDED":
                    reason = "동점(UNDECIDED) 처리 규칙에 따라 FAIL"
                else:
                    reason = f"판정 불일치 (판정: {verdict}, 기대값: {expected_label})"
                fail_cases.append((case_key, reason))
                print(f"판정: {verdict} | expected: {expected_label} | FAIL ({reason})")

        except Exception as exc:
            reason = f"스키마/크기 검증 오류: {exc}"
            fail_cases.append((case_key, reason))
            print(f"판정: ERROR | FAIL ({reason})")

    # 성능 분석 (3×3 포함, 5×5 / 13×13 / 25×25 출력)
    target_sizes = [3, 5, 13, 25]
    for sz in target_sizes:
        if sz not in perf_samples:
            perf_samples[sz] = generate_synthetic_patterns(sz)

    print_performance_table(target_sizes, perf_samples, repeats=REPEAT_COUNT)

    print("\n#---------------------------------------")
    print("# [4] 결과 요약")
    print("#---------------------------------------")
    print(f"총 테스트: {total_count}개")
    print(f"통과: {pass_count}개")
    print(f"실패: {len(fail_cases)}개")

    if fail_cases:
        print("\n실패 케이스:")
        for cid, rsn in fail_cases:
            print(f"- {cid}: {rsn}")
    else:
        print("\n실패 케이스: 없음 (라벨 정규화 및 허용오차 정책 적용 완료)")


def run_bonus_optimization_mode() -> None:
    """[보너스] 패턴 자동 생성기 및 2D vs 1D 메모리 접근 최적화 성능 비교"""
    print("\n#---------------------------------------")
    print("# [보너스] 패턴 생성기 & 1D 메모리 최적화 비교")
    print("#---------------------------------------")
    raw_n = input("생성할 패턴 크기 N 입력 (예: 25, 50, 100): ").strip()
    if not raw_n.isdigit() or int(raw_n) <= 0:
        print("입력 오류: 1 이상의 정수를 입력하세요.")
        return

    n = int(raw_n)
    cross_pat, x_pat = generate_synthetic_patterns(n)
    cross_1d = cross_pat.to_1d()
    x_1d = x_pat.to_1d()

    repeats = 100
    t0 = time.perf_counter()
    for _ in range(repeats):
        compute_mac(cross_pat, x_pat)
    ms_2d = ((time.perf_counter() - t0) / repeats) * 1000.0

    t1 = time.perf_counter()
    for _ in range(repeats):
        compute_mac_1d(cross_1d, x_1d)
    ms_1d = ((time.perf_counter() - t1) / repeats) * 1000.0

    print(f"\n✓ {n}×{n} Cross / X 패턴 자동 생성 완료 (연산 횟수 N² = {n * n})")
    print(f"- 2차원 이중 반복문 평균 시간 ({repeats}회): {ms_2d:.5f} ms")
    print(f"- 1차원 단일 반복문 평균 시간 ({repeats}회): {ms_1d:.5f} ms")


def main() -> None:
    while True:
        print("\n=== Mini NPU Simulator ===")
        print("[모드 선택]")
        print("1. 사용자 입력 (3x3)")
        print("2. data.json 분석")
        print("3. [보너스] 패턴 생성 및 1D 최적화 비교")
        print("0. 종료")
        choice = input("선택: ").strip()

        if choice == "1":
            run_user_input_mode()
        elif choice == "2":
            run_json_analysis_mode("data.json")
        elif choice == "3":
            run_bonus_optimization_mode()
        elif choice == "0":
            print("프로그램을 종료합니다.")
            break
        else:
            print("올바른 메뉴 번호(0, 1, 2, 3)를 선택해 주세요.")


if __name__ == "__main__":
    main()