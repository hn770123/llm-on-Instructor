"""
2段階応答パターンの実装例

このファイルでは、Instructorフレームワークを使った様々な2段階応答パターンを
実装しています。小規模LLM（Ollama + llama3.1:8b）での使用を想定しています。
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Any, Dict
import instructor
import ollama
from datetime import date


# =============================================================================
# パターン1: Chain of Thought (2段階)
# =============================================================================

class ThinkingProcess(BaseModel):
    """第1段階: 推論プロセス（自然言語）"""
    reasoning: str = Field(
        description="問題を解決するための段階的な推論プロセスを自然な日本語で詳しく記述してください"
    )
    intermediate_steps: List[str] = Field(
        description="推論の各ステップを箇条書きで記述"
    )


class ChainOfThoughtResponse(BaseModel):
    """第2段階: 構造化された最終応答"""
    final_answer: str = Field(
        description="推論後の最終的な答え"
    )
    confidence: float = Field(
        description="答えの確信度 (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    reasoning_summary: str = Field(
        description="推論プロセスの要約"
    )


def example_basic_cot():
    """2段階Chain of Thoughtの例"""
    client = instructor.from_openai(
        ollama.Client(),
        mode=instructor.Mode.JSON
    )

    # 第1段階: 推論プロセスを生成
    print("=== Chain of Thought (2段階) ===")
    print("第1段階: 推論プロセスを生成中...")

    thinking = client.chat.completions.create(
        model="llama3.1:8b",
        response_model=ThinkingProcess,
        messages=[
            {
                "role": "system",
                "content": "問題を段階的に考えて、推論プロセスを詳しく記述してください。"
            },
            {
                "role": "user",
                "content": "田中さんは5個のリンゴを持っています。3個食べて、その後友達から4個もらいました。今何個ありますか？"
            }
        ],
    )

    print(f"推論プロセス: {thinking.reasoning}")
    print(f"ステップ: {', '.join(thinking.intermediate_steps)}")

    # 第2段階: 推論から構造化された答えを抽出
    print("\n第2段階: 構造化データを抽出中...")

    response = client.chat.completions.create(
        model="llama3.1:8b",
        response_model=ChainOfThoughtResponse,
        messages=[
            {
                "role": "system",
                "content": "以下の推論プロセスから最終的な答えと確信度を抽出してください。"
            },
            {
                "role": "user",
                "content": f"推論プロセス:\n{thinking.reasoning}\n\nステップ:\n" + "\n".join(f"- {step}" for step in thinking.intermediate_steps)
            }
        ],
    )

    print(f"最終答え: {response.final_answer}")
    print(f"確信度: {response.confidence}")
    print(f"要約: {response.reasoning_summary}")
    print()


# =============================================================================
# パターン2: Chain of Thought with Exclusion
# =============================================================================

class InternalThinkingResponse(BaseModel):
    """内部推論プロセスを除外する応答モデル（小規模LLM推奨）"""

    thinking: str = Field(
        description="自然な日本語で考えるプロセス。専門用語を使わず、人間のように段階的に考えてください",
        exclude=True  # model_dump()時に除外される
    )
    answer: str = Field(
        description="最終的な答え"
    )
    confidence: float = Field(
        description="確信度 (0.0-1.0)",
        ge=0.0,
        le=1.0
    )


def example_cot_with_exclusion():
    """内部推論を除外するパターンの例"""
    client = instructor.from_openai(
        ollama.Client(),
        mode=instructor.Mode.JSON
    )

    response = client.chat.completions.create(
        model="llama3.1:8b",
        response_model=InternalThinkingResponse,
        messages=[
            {
                "role": "user",
                "content": "「Python」という言語の主な特徴を3つ教えてください"
            }
        ],
    )

    print("=== Chain of Thought with Exclusion ===")
    print(f"内部推論（直接アクセス）: {response.thinking}")
    print(f"最終答え: {response.answer}")
    print(f"確信度: {response.confidence}")
    print()
    print("model_dump()の結果（excludeされたフィールドは含まれない）:")
    print(response.model_dump())
    print()


# =============================================================================
# パターン3: Tab-CoT (Tabular Chain of Thought)
# =============================================================================

class ReasoningStep(BaseModel):
    """個別の推論ステップ"""
    step: int = Field(description="ステップ番号")
    subquestion: str = Field(description="このステップで答えるべき小問題")
    procedure: str = Field(description="実行する手順")
    result: str = Field(description="このステップの結果")


class TabularCoTResponse(BaseModel):
    """構造化された推論プロセスを持つ応答"""
    reasoning: List[ReasoningStep] = Field(
        description="段階的な推論ステップのリスト。各ステップは明確な小問題と結果を持つ"
    )
    final_answer: str = Field(
        description="すべての推論ステップを経た後の最終的な答え"
    )


def example_tabular_cot():
    """Tab-CoT（構造化された推論テーブル）の例"""
    client = instructor.from_openai(
        ollama.Client(),
        mode=instructor.Mode.JSON
    )

    response = client.chat.completions.create(
        model="llama3.1:8b",
        response_model=TabularCoTResponse,
        messages=[
            {
                "role": "system",
                "content": "段階的な推論ステップを明確な構造で示してください"
            },
            {
                "role": "user",
                "content": "1から100までの偶数の合計はいくつですか？"
            }
        ],
    )

    print("=== Tabular Chain of Thought ===")
    for step in response.reasoning:
        print(f"ステップ {step.step}: {step.subquestion}")
        print(f"  手順: {step.procedure}")
        print(f"  結果: {step.result}")
    print(f"\n最終答え: {response.final_answer}")
    print()


# =============================================================================
# パターン4: Maybe Pattern (2段階 - 自然言語フォールバック)
# =============================================================================

class TextAnalysis(BaseModel):
    """第1段階: テキスト分析"""
    contains_user_info: bool = Field(description="ユーザー情報が含まれているか")
    analysis: str = Field(description="テキスト内容の分析")
    extracted_elements: List[str] = Field(description="見つかった情報要素のリスト")


class UserDetail(BaseModel):
    """ユーザー詳細情報"""
    name: str = Field(description="ユーザーの名前")
    age: int = Field(description="年齢", ge=0, le=150)
    email: str = Field(description="メールアドレス")
    occupation: Optional[str] = Field(default=None, description="職業")


class MaybeUserResponse(BaseModel):
    """第2段階: ユーザー情報の抽出結果（失敗時の自然言語メッセージ付き）"""
    result: Optional[UserDetail] = Field(
        default=None,
        description="抽出されたユーザー詳細情報。情報が不足している場合はNone"
    )
    error: bool = Field(
        default=False,
        description="抽出に失敗したかどうか"
    )
    message: Optional[str] = Field(
        default=None,
        description="エラーまたは説明のための自然言語メッセージ。失敗時には何が不足しているかを説明"
    )


def example_maybe_pattern():
    """2段階Maybeパターン（自然言語フォールバック）の例"""
    client = instructor.from_openai(
        ollama.Client(),
        mode=instructor.Mode.JSON
    )

    # 成功ケース
    print("=== Maybe Pattern (2段階) - 成功ケース ===")
    print("第1段階: テキスト分析中...")

    analysis1 = client.chat.completions.create(
        model="llama3.1:8b",
        response_model=TextAnalysis,
        messages=[
            {
                "role": "system",
                "content": "テキストを分析して、ユーザー情報が含まれているか判定してください。"
            },
            {
                "role": "user",
                "content": "田中太郎は35歳のエンジニアで、メールアドレスはtanaka@example.comです"
            }
        ],
    )

    print(f"ユーザー情報含有: {analysis1.contains_user_info}")
    print(f"分析: {analysis1.analysis}")
    print(f"抽出要素: {', '.join(analysis1.extracted_elements)}")

    print("\n第2段階: 構造化データを抽出中...")

    response1 = client.chat.completions.create(
        model="llama3.1:8b",
        response_model=MaybeUserResponse,
        messages=[
            {
                "role": "system",
                "content": "分析結果からユーザー情報を抽出してください。情報が不足している場合はerrorをtrueにしてmessageで説明してください。"
            },
            {
                "role": "user",
                "content": f"分析結果:\n{analysis1.analysis}\n\n抽出要素:\n" + "\n".join(f"- {elem}" for elem in analysis1.extracted_elements)
            }
        ],
    )

    if response1.error:
        print(f"エラー: {response1.message}")
    else:
        print(f"名前: {response1.result.name}")
        print(f"年齢: {response1.result.age}")
        print(f"メール: {response1.result.email}")
        print(f"職業: {response1.result.occupation}")
    print()

    # 失敗ケース
    print("=== Maybe Pattern (2段階) - 失敗ケース ===")
    print("第1段階: テキスト分析中...")

    analysis2 = client.chat.completions.create(
        model="llama3.1:8b",
        response_model=TextAnalysis,
        messages=[
            {
                "role": "system",
                "content": "テキストを分析して、ユーザー情報が含まれているか判定してください。"
            },
            {
                "role": "user",
                "content": "今日は良い天気ですね"
            }
        ],
    )

    print(f"ユーザー情報含有: {analysis2.contains_user_info}")
    print(f"分析: {analysis2.analysis}")

    print("\n第2段階: 構造化データを抽出中...")

    response2 = client.chat.completions.create(
        model="llama3.1:8b",
        response_model=MaybeUserResponse,
        messages=[
            {
                "role": "system",
                "content": "分析結果からユーザー情報を抽出してください。情報が不足している場合はerrorをtrueにしてmessageで説明してください。"
            },
            {
                "role": "user",
                "content": f"分析結果:\n{analysis2.analysis}\n\nユーザー情報含有: {analysis2.contains_user_info}"
            }
        ],
    )

    if response2.error:
        print(f"エラー: {response2.message}")
    else:
        print(f"ユーザー情報: {response2.result}")
    print()


# =============================================================================
# パターン5: Self-Correction with Retry
# =============================================================================

class ValidatedUserInfo(BaseModel):
    """バリデーション付きユーザー情報"""
    name: str = Field(description="ユーザーの名前")
    age: int = Field(description="年齢", ge=0, le=150)
    email: str = Field(description="メールアドレス")

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if '@' not in v:
            raise ValueError('有効なメールアドレスではありません。@記号を含む必要があります')
        if '.' not in v.split('@')[1]:
            raise ValueError('メールアドレスのドメイン部分にドットが必要です')
        return v

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if len(v) < 2:
            raise ValueError('名前は2文字以上である必要があります')
        return v


def example_self_correction():
    """Self-Correction（自己修正）の例"""
    client = instructor.from_openai(
        ollama.Client(),
        mode=instructor.Mode.JSON
    )

    print("=== Self-Correction with Retry ===")

    # 意図的にエラーを含むプロンプト（@がないメールアドレス）
    try:
        response = client.chat.completions.create(
            model="llama3.1:8b",
            response_model=ValidatedUserInfo,
            messages=[
                {
                    "role": "system",
                    "content": "ユーザー情報を正確に抽出してください"
                },
                {
                    "role": "user",
                    "content": "太郎は25歳で、メールはtaro.example.comです"
                }
            ],
            max_retries=3  # 最大3回まで再試行
        )

        print("成功:")
        print(f"名前: {response.name}")
        print(f"年齢: {response.age}")
        print(f"メール: {response.email}")

    except Exception as e:
        print(f"リトライ上限に達しました: {e}")

    print()


# =============================================================================
# パターン6: Plan and Solve (2段階)
# =============================================================================

class PlanStep(BaseModel):
    """計画の1ステップ"""
    step_number: int = Field(description="ステップ番号")
    description: str = Field(description="このステップで行うこと")
    expected_outcome: str = Field(description="期待される結果")


class Plan(BaseModel):
    """第1段階: 実行計画"""
    goal: str = Field(description="達成したい目標")
    steps: List[PlanStep] = Field(description="目標達成のための実行ステップのリスト")
    considerations: List[str] = Field(description="計画時の考慮事項")


class PlanAndSolveResponse(BaseModel):
    """第2段階: 計画に基づいた解決策"""
    execution_summary: str = Field(description="計画に基づいた実行結果の要約")
    step_results: List[str] = Field(description="各ステップの実行結果")
    final_answer: str = Field(description="最終的な答え")
    confidence: float = Field(description="答えの確信度", ge=0.0, le=1.0)


def example_plan_and_solve():
    """2段階Plan and Solve（計画→実行）の例"""
    client = instructor.from_openai(
        ollama.Client(),
        mode=instructor.Mode.JSON
    )

    print("=== Plan and Solve (2段階) ===")
    print("第1段階: 計画を立案中...")

    # 第1段階: 計画を立てる
    plan = client.chat.completions.create(
        model="llama3.1:8b",
        response_model=Plan,
        messages=[
            {
                "role": "system",
                "content": "問題を解決するための詳細な計画を立ててください"
            },
            {
                "role": "user",
                "content": "3日間の東京旅行の予算を計算してください。宿泊費は1泊1万円、食費は1日5千円、交通費は1日2千円です"
            }
        ],
    )

    print(f"目標: {plan.goal}")
    print("\n計画:")
    for step in plan.steps:
        print(f"{step.step_number}. {step.description}")
        print(f"   期待結果: {step.expected_outcome}")
    print(f"\n考慮事項: {', '.join(plan.considerations)}")

    # 第2段階: 計画を実行して答えを出す
    print("\n第2段階: 計画を実行中...")

    response = client.chat.completions.create(
        model="llama3.1:8b",
        response_model=PlanAndSolveResponse,
        messages=[
            {
                "role": "system",
                "content": "以下の計画に従って問題を解決してください"
            },
            {
                "role": "user",
                "content": f"計画:\n目標: {plan.goal}\n\nステップ:\n" + "\n".join(f"{step.step_number}. {step.description} (期待結果: {step.expected_outcome})" for step in plan.steps)
            }
        ],
    )

    print(f"実行要約: {response.execution_summary}")
    print("\nステップ結果:")
    for i, result in enumerate(response.step_results, 1):
        print(f"{i}. {result}")
    print(f"\n最終答え: {response.final_answer}")
    print(f"確信度: {response.confidence}")
    print()


# =============================================================================
# パターン7: Flexible Natural Language First
# =============================================================================

class FlexibleResponse(BaseModel):
    """柔軟な応答モデル - 自然言語優先"""

    natural_response: str = Field(
        description="自然な日本語での完全な回答。ユーザーが読みやすい形で答えてください"
    )

    structured_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="自然言語応答から抽出できる場合のみ、構造化データを提供してください"
    )

    extraction_confidence: float = Field(
        description="構造化データの抽出確信度 (0.0-1.0)",
        ge=0.0,
        le=1.0
    )


def example_flexible_natural_first():
    """自然言語優先の柔軟な応答の例"""
    client = instructor.from_openai(
        ollama.Client(),
        mode=instructor.Mode.JSON
    )

    response = client.chat.completions.create(
        model="llama3.1:8b",
        response_model=FlexibleResponse,
        messages=[
            {
                "role": "user",
                "content": "Pythonのリスト内包表記について簡単に説明してください"
            }
        ],
    )

    print("=== Flexible Natural Language First ===")
    print(f"自然言語応答:\n{response.natural_response}")
    print(f"\n構造化データ: {response.structured_data}")
    print(f"抽出確信度: {response.extraction_confidence}")
    print()


# =============================================================================
# パターン8: Two-Stage Explicit (2回呼び出し)
# =============================================================================

class NaturalLanguageOnly(BaseModel):
    """第1段階: 自然言語のみの応答"""
    answer: str = Field(description="自然な日本語での完全な回答")
    key_points: List[str] = Field(description="回答の要点リスト")


class ExtractedStructure(BaseModel):
    """第2段階: 構造化データ"""
    summary: str = Field(description="要約")
    main_topic: str = Field(description="主要トピック")
    details: Dict[str, str] = Field(description="詳細情報")


def example_two_stage_explicit():
    """明示的な2段階呼び出しの例"""
    client = instructor.from_openai(
        ollama.Client(),
        mode=instructor.Mode.JSON
    )

    print("=== Two-Stage Explicit (2回呼び出し) ===")

    # ステップ1: 自然言語で応答を生成
    print("ステップ1: 自然言語応答を生成中...")
    natural_resp = client.chat.completions.create(
        model="llama3.1:8b",
        response_model=NaturalLanguageOnly,
        messages=[
            {
                "role": "system",
                "content": "自然で読みやすい日本語で答えてください"
            },
            {
                "role": "user",
                "content": "機械学習とディープラーニングの違いを説明してください"
            }
        ],
    )

    print(f"自然言語応答:\n{natural_resp.answer}")
    print(f"要点: {', '.join(natural_resp.key_points)}")

    # ステップ2: 自然言語応答を構造化データに変換
    print("\nステップ2: 構造化データに変換中...")
    structured_resp = client.chat.completions.create(
        model="llama3.1:8b",
        response_model=ExtractedStructure,
        messages=[
            {
                "role": "system",
                "content": "以下のテキストから構造化された情報を抽出してください"
            },
            {
                "role": "user",
                "content": f"以下のテキストから情報を抽出してください:\n\n{natural_resp.answer}"
            }
        ],
    )

    print(f"要約: {structured_resp.summary}")
    print(f"主要トピック: {structured_resp.main_topic}")
    print("詳細情報:")
    for key, value in structured_resp.details.items():
        print(f"  {key}: {value}")
    print()


# =============================================================================
# メイン実行
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Instructor 2段階応答パターン - 実装例")
    print("=" * 80)
    print()

    # 各パターンの実行例
    # 注: 実際の実行にはOllamaとllama3.1:8bモデルが必要です

    try:
        # パターン1: 基本的なChain of Thought
        example_basic_cot()

        # パターン2: 内部推論の除外
        example_cot_with_exclusion()

        # パターン3: 構造化された推論テーブル
        example_tabular_cot()

        # パターン4: Maybe（自然言語フォールバック）
        example_maybe_pattern()

        # パターン5: Self-Correction
        example_self_correction()

        # パターン6: Plan and Solve
        example_plan_and_solve()

        # パターン7: 自然言語優先
        example_flexible_natural_first()

        # パターン8: 2段階明示的呼び出し
        example_two_stage_explicit()

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("\n注意: この例を実行するには以下が必要です:")
        print("1. Ollamaがインストールされ、実行中であること")
        print("2. llama3.1:8bモデルがダウンロード済みであること")
        print("   コマンド: ollama pull llama3.1:8b")
        print("3. 必要なPythonパッケージがインストール済みであること")
        print("   コマンド: pip install instructor ollama pydantic")
