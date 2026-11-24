# ==============================================================================
# [프로그램 전체 소개] Mini NPU Simulator (그림 판별 숫자 계산기)
#
# - 사람이 눈으로 '+'(십자가)와 'X'를 구별하듯, 눈이 없는 컴퓨터가
#   0(빈칸)과 1(색칠된 칸)로 이루어진 '숫자 바둑판' 점수를 계산해 모양을 맞히는 프로그램입니다.
# - 핵심 원리(MAC 연산): 두 바둑판을 겹쳐서 같은 자리에 있는 숫자끼리 곱한 뒤 모두 더합니다.
# ==============================================================================

import json     # 문제 파일(data.json)을 읽어오기 위한 파이썬 기본 도구
import re       # 'size_5_1' 같은 글자에서 숫자(5)만 쏙 뽑아내기 위한 기본 도구
import time     # 계산하는 데 몇 밀리초(ms)가 걸리는지 초시계처럼 재는 기본 도구
from pathlib import Path  # 파일이 폴더 안에 실제로 존재하는지 확인하는 기본 도구

# [기본 설정값]
# 컴퓨터가 소수점 계산을 할 때 생기는 아주 미세한 오차(0.000000001 미만)는
# 사실상 점수가 똑같은 '무승부'로 처리하기 위한 기준값입니다.
EPSILON = 1e-9

# 계산 속도를 정확히 재기 위해 같은 계산을 몇 번 반복해서 평균을 낼지 정하는 횟수입니다.
REPEAT_COUNT = 10


class PatternMatrix:
    """
    [숫자 바둑판 보관함]
    가로 N칸, 세로 N칸 크기의 2차원 숫자 바둑판(정답지 필터 또는 문제 패턴)을
    저장하고, 특정 칸의 숫자를 읽거나 고치는 역할을 담당합니다.
    """

    def __init__(self, size: int, grid=None):
        # 바둑판 크기가 0 이하이면 잘못된 크기이므로 안내 문구를 띄웁니다.
        if size <= 0:
            raise ValueError("배열 크기는 1 이상이어야 합니다.")
        self.size = size

        # 미리 전달받은 숫자 표가 없으면, 모든 칸이 0.0으로 채워진 빈 바둑판을 만듭니다.
        if grid is None:
            self.grid = [[0.0 for _ in range(size)] for _ in range(size)]
        else:
            # 전달받은 표의 가로·세로 줄 수가 약속된 크기(size×size)와 다르면 불량 데이터로 걸러냅니다.
            if len(grid) != size or any(len(row) != size for row in grid):
                raise ValueError(f"크기 불일치: {size}×{size} 정사각 배열이 아닙니다.")
            # 모든 칸의 숫자를 소수점 계산이 가능한 숫자(float) 형태로 깔끔하게 저장합니다.
            self.grid = [[float(val) for val in row] for row in grid]

    def get_value(self, row: int, col: int) -> float:
        """바둑판의 특정 위치(row번째 줄, col번째 칸)에 적힌 숫자를 읽어옵니다."""
        self._check_bounds(row, col)
        return self.grid[row][col]

    def set_value(self, row: int, col: int, value: float) -> None:
        """바둑판의 특정 위치(row번째 줄, col번째 칸)에 새로운 숫자를 적어 넣습니다."""
        self._check_bounds(row, col)
        self.grid[row][col] = float(value)

    def _check_bounds(self, row: int, col: int) -> None:
        """바둑판 범위를 벗어난 엉뚱한 칸을 찾으려고 할 때 미리 막아주는 안전장치입니다."""
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise IndexError(f"인덱스 범위 초과: ({row}, {col})는 {self.size}×{self.size} 범위를 벗어났습니다.")

    def to_1d(self) -> list:
        """
        [보너스 기능] 2차원 네모 바둑판을 한 줄짜리 긴 숫자 목록(길이 N²)으로 쭉 펴줍니다.
        컴퓨터가 칸을 하나씩 찾아가는 과정을 단순하게 만들어 속도를 높일 때 사용합니다.
        """
        flat = []
        for row in self.grid:
            flat.extend(row)
        return flat


def normalize_label(raw_label: str) -> str:
    """
    [이름표 통일 기능]
    문제 파일(data.json) 안에 십자가가 '+' 또는 'cross'처럼 제각각 적혀 있어도
    프로그램 안에서는 항상 표준 이름인 'Cross'와 'X'로 통일해 줍니다.
    """
    if not isinstance(raw_label, str):
        raise ValueError(f"유효하지 않은 라벨 형식: {raw_label}")
    cleaned = raw_label.strip().lower()  # 앞뒤 공백을 지우고 소문자로 바꿉니다.
    if cleaned in ("+", "cross"):
        return "Cross"
    if cleaned == "x":
        return "X"
    raise ValueError(f"지원하지 않는 라벨 값입니다: '{raw_label}'")


def compute_mac(pattern: PatternMatrix, filter_matrix: PatternMatrix) -> float:
    """
    [핵심 계산 기능: MAC 연산]
    문제 바둑판(pattern)과 정답지 바둑판(filter_matrix)을 똑같이 겹쳐 놓고,
    같은 칸에 있는 숫자끼리 곱한 뒤 그 결과를 전부 더해서 '닮은 점수'를 계산합니다.
    (외부 계산 도구 없이 파이썬 기본 반복문만으로 직접 계산합니다.)
    """
    if pattern.size != filter_matrix.size:
        raise ValueError(
            f"패턴 크기({pattern.size}×{pattern.size})와 필터 크기({filter_matrix.size}×{filter_matrix.size})가 일치하지 않습니다."
        )
    n = pattern.size
    p_grid = pattern.grid
    f_grid = filter_matrix.grid
    total = 0.0  # 점수를 누적해서 더해 나갈 바구니

    # 위에서 아래로 한 줄씩(r), 왼쪽에서 오른쪽으로 한 칸씩(c) 이동하며 곱하고 더합니다.
    for r in range(n):
        for c in range(n):
            total += p_grid[r][c] * f_grid[r][c]
    return float(total)


def compute_mac_1d(pattern_1d: list, filter_1d: list) -> float:
    """
    [보너스 기능: 1줄 방식 MAC 연산]
    한 줄로 길게 편 두 숫자 목록을 처음부터 끝까지 한 번만 훑으면서 곱하고 더합니다.
    """
    if len(pattern_1d) != len(filter_1d):
        raise ValueError("1차원 배열 길이가 일치하지 않습니다.")
    total = 0.0
    for idx in range(len(pattern_1d)):
        total += pattern_1d[idx] * filter_1d[idx]
    return float(total)


def judge_scores(score_a: float, score_b: float, label_a: str, label_b: str, tie_label: str = "UNDECIDED") -> str:
    """
    [최종 승자 판정 기능]
    두 정답지(A와 B, 또는 Cross와 X)의 점수를 비교해 더 높은 쪽을 정답으로 고릅니다.
    단, 두 점수 차이가 허용오차(1e-9)보다 작으면 사실상 같은 점수로 보고 '무승부'로 판정합니다.
    """
    if abs(score_a - score_b) < EPSILON:
        return tie_label
    return label_a if score_a > score_b else label_b


def measure_mac_avg_ms(pattern: PatternMatrix, filter_matrix: PatternMatrix, repeats: int = REPEAT_COUNT) -> float:
    """
    [순수 계산 속도 측정기]
    글자를 입력하거나 파일을 읽는 시간은 빼고, 오직 곱하고 더하는 계산(compute_mac)만
    10번 반복 실행한 뒤 1번당 평균 몇 밀리초(ms)가 걸렸는지 계산합니다.
    """
    start = time.perf_counter()  # 초시계 시작
    for _ in range(repeats):
        compute_mac(pattern, filter_matrix)
    elapsed_sec = time.perf_counter() - start  # 초시계 정지 후 총 걸린 시간 계산
    return (elapsed_sec / repeats) * 1000.0    # 평균 시간(초)을 밀리초(ms) 단위로 변환


def generate_synthetic_patterns(n: int) -> tuple:
    """
    [보너스 기능: 연습용 바둑판 자동 생성기]
    원하는 크기(n)를 알려주면 정가운데 가로·세로 줄이 1인 '십자가(Cross)' 바둑판과
    양쪽 대각선 줄이 1인 'X' 바둑판을 자동으로 만들어 줍니다.
    """
    cross = PatternMatrix(n)
    x_pat = PatternMatrix(n)
    mid = n // 2  # 바둑판의 정가운데 줄 번호
    for r in range(n):
        for c in range(n):
            if r == mid or c == mid:
                cross.set_value(r, c, 1.0)
            if r == c or (r + c) == (n - 1):
                x_pat.set_value(r, c, 1.0)
    return cross, x_pat

def read_3x3_matrix(title: str) -> PatternMatrix:
    """
    [사용자 직접 입력 도우미 (모드 1용)]
    키보드로 3×3 바둑판 숫자를 한 줄에 3개씩 총 3줄 입력받습니다.
    숫자가 아닌 글자를 치거나 3칸을 안 맞추면 안내 문구를 보여주고 그 줄만 다시 입력받습니다.
    """
    print(f"{title} (3줄 입력, 공백 구분)")
    rows = []
    row_idx = 0
    while row_idx < 3:
        raw_line = input().strip()
        parts = raw_line.split()  # 띄어쓰기를 기준으로 숫자를 나눕니다.
        if len(parts) != 3:
            print("입력 형식 오류: 각 줄에 3개의 숫자를 공백으로 구분해 입력하세요.")
            continue
        try:
            nums = [float(x) for x in parts]  # 글자를 실제 계산 가능한 숫자로 바꿉니다.
        except ValueError:
            print("입력 형식 오류: 각 줄에 3개의 숫자를 공백으로 구분해 입력하세요.")
            continue
        rows.append(nums)
        row_idx += 1
    return PatternMatrix(3, rows)


def print_performance_table(sizes: list, sample_pairs: dict, repeats: int = REPEAT_COUNT) -> None:
    """
    [속도 비교 표 출력기]
    바둑판 크기(3×3, 5×5, 13×13, 25×25)별로 평균 계산 시간(ms)과
    총 계산 칸 수(N²)를 보기 좋은 표 형태로 화면에 출력합니다.
    """
    print(f"\n#---------------------------------------")
    print(f"# [성능 분석] (평균/{repeats}회)")
    print(f"#---------------------------------------")
    print(f"{'크기':<10} {'평균 시간(ms)':<16} {'연산 횟수':<10}")
    print("-" * 38)
    for n in sizes:
        pat, flt = sample_pairs[n]
        avg_ms = measure_mac_avg_ms(pat, flt, repeats=repeats)
        ops = n * n  # 가로 N칸 × 세로 N칸 = 총 계산 칸 수(N²)
        size_label = f"{n}×{n}"
        print(f"{size_label:<10} {avg_ms:<16.4f} {ops:<10}")


def run_user_input_mode() -> None:
    """
    [메뉴 1번 실행 흐름: 내가 직접 3×3 바둑판 입력해서 판정하기]
    정답지 A와 B를 입력받고, 문제 바둑판을 입력받아 점수와 승자를 보여줍니다.
    """
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

    # A 점수와 B 점수를 각각 계산하고 승자를 가립니다.
    score_a = compute_mac(pattern, filter_a)
    score_b = compute_mac(pattern, filter_b)
    avg_ms = measure_mac_avg_ms(pattern, filter_a, repeats=REPEAT_COUNT)
    verdict = judge_scores(score_a, score_b, "A", "B", tie_label="판정 불가")

    print("\n#---------------------------------------")
    print("# [3] MAC 결과")
    print("#---------------------------------------")
    if verdict == "판정 불가":
        # 무승부일 때는 미세한 소수점 차이까지 확인할 수 있게 소수점 16자리까지 보여줍니다.
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
    """
    [메뉴 2번 실행 흐름: 문제 파일(data.json) 한꺼번에 채점하기]
    1) 파일에서 크기별 정답지(필터)를 읽어와 이름표를 'Cross', 'X'로 통일합니다.
    2) 문제들을 하나씩 채점해 기대 정답과 비교(PASS/FAIL)합니다.
    3) 크기별 속도 비교 표와 최종 성적표(총/통과/실패)를 출력합니다.
    """
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

    # [1단계] 크기별 정답지(필터) 불러오기
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
                std_label = normalize_label(raw_k)  # 'cross' -> 'Cross', 'x' -> 'X' 통일
                normalized_group[std_label] = PatternMatrix(n, matrix_data)
            if "Cross" not in normalized_group or "X" not in normalized_group:
                raise ValueError("필수 필터(Cross, X)가 누락되었습니다.")
            loaded_filters[n] = normalized_group
            print(f"✓ {size_key:<7} 필터 로드 완료 (Cross, X)")
        except Exception as exc:
            print(f"✗ {size_key:<7} 필터 로드 실패: {exc}")

    # [2단계] 문제 하나씩 채점하고 PASS/FAIL 판정하기
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

            # 문제 이름('size_5_1')에서 숫자 '5'를 뽑아내어 5×5 정답지를 찾아냅니다.
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

            # 바둑판 크기 검증 및 정답 이름표('+' -> 'Cross') 통일
            pattern_matrix = PatternMatrix(n, raw_input)
            expected_label = normalize_label(raw_expected)

            cross_filter = loaded_filters[n]["Cross"]
            x_filter = loaded_filters[n]["X"]

            if n not in perf_samples:
                perf_samples[n] = (pattern_matrix, cross_filter)

            # Cross 점수와 X 점수를 각각 계산하고 최종 판정을 내립니다.
            cross_score = compute_mac(pattern_matrix, cross_filter)
            x_score = compute_mac(pattern_matrix, x_filter)
            verdict = judge_scores(cross_score, x_score, "Cross", "X", tie_label="UNDECIDED")

            print(f"Cross 점수: {cross_score:.1f}" if cross_score.is_integer() else f"Cross 점수: {cross_score:.16f}")
            print(f"X 점수: {x_score:.1f}" if x_score.is_integer() else f"X 점수: {x_score:.16f}")

            # 판정 결과가 기대 정답과 같으면 PASS, 다르면 FAIL로 기록합니다.
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
            # 크기가 안 맞는 불량 문제가 들어와도 프로그램이 멈추지 않고 이 문제만 FAIL 처리합니다.
            reason = f"스키마/크기 검증 오류: {exc}"
            fail_cases.append((case_key, reason))
            print(f"판정: ERROR | FAIL ({reason})")

    # [3단계] 크기별 계산 속도 비교 표 출력 (3×3, 5×5, 13×13, 25×25)
    target_sizes = [3, 5, 13, 25]
    for sz in target_sizes:
        if sz not in perf_samples:
            perf_samples[sz] = generate_synthetic_patterns(sz)

    print_performance_table(target_sizes, perf_samples, repeats=REPEAT_COUNT)

    # [4단계] 최종 성적표 요약 출력
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
    """
    [메뉴 3번 보너스 기능: 바둑판 자동 생성 및 2차원 vs 1줄 방식 속도 비교]
    원하는 크기의 바둑판을 자동으로 만든 뒤, 기존 방식과 한 줄로 편 방식의 속도를 비교합니다.
    """
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
    """
    [프로그램 메인 시작점]
    사용자가 '0. 종료'를 누를 때까지 메뉴 화면을 계속 보여주며 선택한 기능을 실행합니다.
    """
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


# 이 파일을 직접 실행했을 때 main() 메뉴 화면을 띄우도록 시작해 주는 줄입니다.
if __name__ == "__main__":
    main()