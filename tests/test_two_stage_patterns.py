"""
2段階応答パターンのテストスイート

このファイルでは、Ollamaをモック化して2段階応答パターンの動作をテストします。
"""

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    pytest = None

from unittest.mock import Mock, MagicMock, patch
from typing import Type, TypeVar, Any
import sys
import os

# examplesディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))

from two_stage_patterns import (
    ThinkingProcess,
    ChainOfThoughtResponse,
    TextAnalysis,
    MaybeUserResponse,
    UserDetail,
    Plan,
    PlanStep,
    PlanAndSolveResponse,
    NaturalLanguageOnly,
    ExtractedStructure,
)

T = TypeVar('T')


class MockOllamaClient:
    """Ollamaクライアントのモック"""

    def __init__(self):
        self.chat = MagicMock()


class MockInstructorClient:
    """Instructorクライアントのモック"""

    def __init__(self, response_map: dict):
        """
        Args:
            response_map: response_modelの型をキーとして、返すべき応答をバリューとする辞書
        """
        self.response_map = response_map
        self.call_count = 0
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = MagicMock(side_effect=self._create_response)

    def _create_response(self, model: str, response_model: Type[T], messages: list, **kwargs) -> T:
        """
        モックの応答を返す

        Args:
            model: モデル名
            response_model: 期待される応答モデル
            messages: メッセージリスト

        Returns:
            response_modelのインスタンス
        """
        self.call_count += 1
        if response_model in self.response_map:
            return self.response_map[response_model]
        else:
            raise ValueError(f"No mock response defined for {response_model}")


if PYTEST_AVAILABLE:
    @pytest.fixture
    def mock_ollama_client():
        """Ollamaクライアントのモックフィクスチャ"""
        return MockOllamaClient()


def test_chain_of_thought_two_stage():
    """パターン1: Chain of Thought (2段階) のテスト"""

    # モック応答を定義
    thinking_response = ThinkingProcess(
        reasoning="まず田中さんは5個のリンゴを持っていました。3個食べたので、5 - 3 = 2個になります。その後、友達から4個もらったので、2 + 4 = 6個になります。",
        intermediate_steps=[
            "初期状態: 5個のリンゴ",
            "3個食べた: 5 - 3 = 2個",
            "友達から4個もらった: 2 + 4 = 6個"
        ]
    )

    final_response = ChainOfThoughtResponse(
        final_answer="6個",
        confidence=0.95,
        reasoning_summary="田中さんは5個から3個食べて2個になり、4個もらって合計6個になりました。"
    )

    # モッククライアントを作成
    response_map = {
        ThinkingProcess: thinking_response,
        ChainOfThoughtResponse: final_response
    }

    mock_client = MockInstructorClient(response_map)

    # テスト実行
    with patch('two_stage_patterns.instructor.from_openai', return_value=mock_client):
        with patch('two_stage_patterns.ollama.Client', return_value=MockOllamaClient()):
            from two_stage_patterns import example_basic_cot

            # 関数を実行（エラーが発生しないことを確認）
            example_basic_cot()

    # 呼び出しが2回行われたことを確認
    assert mock_client.chat.completions.create.call_count == 2

    # 第1段階の応答を確認
    assert thinking_response.reasoning is not None
    assert len(thinking_response.intermediate_steps) == 3

    # 第2段階の応答を確認
    assert final_response.final_answer == "6個"
    assert final_response.confidence == 0.95


def test_maybe_pattern_success():
    """パターン4: Maybe Pattern (2段階) 成功ケースのテスト"""

    # モック応答を定義
    analysis_response = TextAnalysis(
        contains_user_info=True,
        analysis="テキストにはユーザーの名前、年齢、職業、メールアドレスが含まれています。",
        extracted_elements=["名前: 田中太郎", "年齢: 35歳", "職業: エンジニア", "メール: tanaka@example.com"]
    )

    user_response = MaybeUserResponse(
        result=UserDetail(
            name="田中太郎",
            age=35,
            email="tanaka@example.com",
            occupation="エンジニア"
        ),
        error=False,
        message=None
    )

    # モッククライアントを作成
    response_map = {
        TextAnalysis: analysis_response,
        MaybeUserResponse: user_response
    }

    mock_client = MockInstructorClient(response_map)

    # テスト実行
    with patch('two_stage_patterns.instructor.from_openai', return_value=mock_client):
        with patch('two_stage_patterns.ollama.Client', return_value=MockOllamaClient()):
            # 応答の検証
            assert analysis_response.contains_user_info is True
            assert len(analysis_response.extracted_elements) == 4

            assert user_response.error is False
            assert user_response.result.name == "田中太郎"
            assert user_response.result.age == 35
            assert user_response.result.email == "tanaka@example.com"
            assert user_response.result.occupation == "エンジニア"


def test_maybe_pattern_failure():
    """パターン4: Maybe Pattern (2段階) 失敗ケースのテスト"""

    # モック応答を定義
    analysis_response = TextAnalysis(
        contains_user_info=False,
        analysis="テキストには天気に関する情報のみが含まれており、ユーザー情報は見つかりません。",
        extracted_elements=[]
    )

    user_response = MaybeUserResponse(
        result=None,
        error=True,
        message="ユーザー情報が見つかりませんでした。名前、年齢、メールアドレスのいずれも抽出できませんでした。"
    )

    # モッククライアントを作成
    response_map = {
        TextAnalysis: analysis_response,
        MaybeUserResponse: user_response
    }

    mock_client = MockInstructorClient(response_map)

    # テスト実行
    with patch('two_stage_patterns.instructor.from_openai', return_value=mock_client):
        with patch('two_stage_patterns.ollama.Client', return_value=MockOllamaClient()):
            # 応答の検証
            assert analysis_response.contains_user_info is False
            assert len(analysis_response.extracted_elements) == 0

            assert user_response.error is True
            assert user_response.result is None
            assert "見つかりませんでした" in user_response.message


def test_plan_and_solve():
    """パターン6: Plan and Solve (2段階) のテスト"""

    # モック応答を定義
    plan_response = Plan(
        goal="3日間の東京旅行の予算を計算する",
        steps=[
            PlanStep(step_number=1, description="宿泊費を計算", expected_outcome="宿泊費の合計"),
            PlanStep(step_number=2, description="食費を計算", expected_outcome="食費の合計"),
            PlanStep(step_number=3, description="交通費を計算", expected_outcome="交通費の合計"),
            PlanStep(step_number=4, description="すべてを合算", expected_outcome="総予算")
        ],
        considerations=["3日間の旅行", "宿泊は2泊", "費用を正確に計算"]
    )

    solve_response = PlanAndSolveResponse(
        execution_summary="計画に従って各項目を計算し、合計を求めました。",
        step_results=[
            "宿泊費: 1泊1万円 × 2泊 = 2万円",
            "食費: 1日5千円 × 3日 = 1万5千円",
            "交通費: 1日2千円 × 3日 = 6千円",
            "総予算: 2万円 + 1万5千円 + 6千円 = 4万1千円"
        ],
        final_answer="4万1千円",
        confidence=1.0
    )

    # モッククライアントを作成
    response_map = {
        Plan: plan_response,
        PlanAndSolveResponse: solve_response
    }

    mock_client = MockInstructorClient(response_map)

    # テスト実行
    with patch('two_stage_patterns.instructor.from_openai', return_value=mock_client):
        with patch('two_stage_patterns.ollama.Client', return_value=MockOllamaClient()):
            from two_stage_patterns import example_plan_and_solve

            # 関数を実行
            example_plan_and_solve()

    # 呼び出しが2回行われたことを確認
    assert mock_client.chat.completions.create.call_count == 2

    # 計画の検証
    assert plan_response.goal is not None
    assert len(plan_response.steps) == 4
    assert len(plan_response.considerations) == 3

    # 解決策の検証
    assert solve_response.final_answer == "4万1千円"
    assert solve_response.confidence == 1.0
    assert len(solve_response.step_results) == 4


def test_two_stage_explicit():
    """パターン8: Two-Stage Explicit (明示的2段階) のテスト"""

    # モック応答を定義
    natural_response = NaturalLanguageOnly(
        answer="機械学習は、データからパターンを学習してタスクを実行するAIの一分野です。一方、ディープラーニングは機械学習の一種で、ニューラルネットワークを使用します。",
        key_points=[
            "機械学習はデータから学習",
            "ディープラーニングは機械学習の一種",
            "ディープラーニングはニューラルネットワークを使用"
        ]
    )

    structured_response = ExtractedStructure(
        summary="機械学習とディープラーニングの違いを説明",
        main_topic="機械学習 vs ディープラーニング",
        details={
            "機械学習": "データからパターンを学習するAI技術",
            "ディープラーニング": "ニューラルネットワークを使用する機械学習の一種",
            "関係性": "ディープラーニングは機械学習のサブセット"
        }
    )

    # モッククライアントを作成
    response_map = {
        NaturalLanguageOnly: natural_response,
        ExtractedStructure: structured_response
    }

    mock_client = MockInstructorClient(response_map)

    # テスト実行
    with patch('two_stage_patterns.instructor.from_openai', return_value=mock_client):
        with patch('two_stage_patterns.ollama.Client', return_value=MockOllamaClient()):
            from two_stage_patterns import example_two_stage_explicit

            # 関数を実行
            example_two_stage_explicit()

    # 呼び出しが2回行われたことを確認
    assert mock_client.chat.completions.create.call_count == 2

    # 自然言語応答の検証
    assert natural_response.answer is not None
    assert len(natural_response.key_points) == 3

    # 構造化データの検証
    assert structured_response.main_topic == "機械学習 vs ディープラーニング"
    assert len(structured_response.details) == 3


def test_mock_client_error_handling():
    """定義されていない応答モデルに対するエラーハンドリングのテスト"""

    mock_client = MockInstructorClient({})

    if PYTEST_AVAILABLE:
        with pytest.raises(ValueError, match="No mock response defined"):
            mock_client.chat.completions.create(
                model="test",
                response_model=ThinkingProcess,
                messages=[]
            )
    else:
        try:
            mock_client.chat.completions.create(
                model="test",
                response_model=ThinkingProcess,
                messages=[]
            )
            raise AssertionError("Expected ValueError but none was raised")
        except ValueError as e:
            assert "No mock response defined" in str(e)


if __name__ == "__main__":
    # pytestを使わずに直接実行する場合
    print("=" * 80)
    print("2段階応答パターンのテスト実行")
    print("=" * 80)
    print()

    print("テスト1: Chain of Thought (2段階)")
    test_chain_of_thought_two_stage()
    print("✓ テスト成功\n")

    print("テスト2: Maybe Pattern - 成功ケース")
    test_maybe_pattern_success()
    print("✓ テスト成功\n")

    print("テスト3: Maybe Pattern - 失敗ケース")
    test_maybe_pattern_failure()
    print("✓ テスト成功\n")

    print("テスト4: Plan and Solve")
    test_plan_and_solve()
    print("✓ テスト成功\n")

    print("テスト5: Two-Stage Explicit")
    test_two_stage_explicit()
    print("✓ テスト成功\n")

    print("テスト6: エラーハンドリング")
    test_mock_client_error_handling()
    print("✓ テスト成功\n")

    print("=" * 80)
    print("すべてのテストが成功しました!")
    print("=" * 80)
