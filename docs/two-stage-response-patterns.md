# LLMの2段階応答パターン with Instructor

## 概要

このドキュメントでは、LLMへのリクエストと応答を2段階にする手法について調査した結果をまとめます。

**調査の背景**:
- 自然言語→自然言語の応答生成が、特に小規模LLMで最も精度が高い
- その後、構造化データに型変換することで、品質と柔軟性を両立したい
- Instructorフレームワークとの統合可能性

## 調査結果サマリー

✅ **結論**: Instructorフレームワークは2段階応答パターンと完全に互換性があります

以下の複数の手法がInstructorで公式にサポートされています：

1. **Chain of Thought (CoT)パターン** - 推論→回答の2段階
2. **Maybeパターン** - 構造化失敗時の自然言語フォールバック
3. **Tab-CoT (Tabular Chain of Thought)** - 構造化された推論ステップ
4. **Self-Correction/Retry** - エラー時の自然言語フィードバックと再試行
5. **Plan and Solve** - 計画→実行の2段階アプローチ

---

## 1. Chain of Thought (CoT) パターン

### 概要
Pydanticモデルに`chain_of_thought`フィールドを追加することで、LLMにまず推論させてから最終的な答えを生成させる手法です。

### パフォーマンス
- GSM8kデータセットで**精度が60%向上**した実績あり
- フィールド名も重要: `"potential_final_choice"` → `"final_answer"` で精度が4.5%から95%に改善

### 実装例

```python
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI

class ReasonedAnswer(BaseModel):
    """推論プロセスを含む回答モデル"""
    chain_of_thought: str = Field(
        description="問題を解決するための段階的な推論プロセス"
    )
    final_answer: str = Field(
        description="推論後の最終的な結論"
    )

client = instructor.from_openai(OpenAI())

response = client.chat.completions.create(
    model="gpt-4",
    response_model=ReasonedAnswer,
    messages=[
        {"role": "user", "content": "5 + 7 × 3 を計算してください"}
    ],
)

print(f"推論: {response.chain_of_thought}")
print(f"答え: {response.final_answer}")
```

### 出力例

```
推論: まず掛け算を先に計算します。7 × 3 = 21。次に足し算を行います。5 + 21 = 26。
答え: 26
```

### 応用: Chain of Thoughtの除外

推論プロセスは内部処理で使いたいが、最終的なAPI出力には含めたくない場合：

```python
class DateRange(BaseModel):
    chain_of_thought: str = Field(
        description="日付範囲を決定した推論",
        exclude=True  # model_dump()時に除外される
    )
    start_date: date
    end_date: date
```

**参考資料**:
- [Text Classification with OpenAI and Pydantic - Instructor](https://python.useinstructor.com/examples/classification/)
- [Customizing Pydantic Models with Field Metadata - Instructor](https://python.useinstructor.com/concepts/fields/)

---

## 2. Tab-CoT (Tabular Chain of Thought)

### 概要
推論を構造化されたテーブル/リストとして出力する手法。より詳細な推論プロセスを追跡できます。

### 実装例

```python
class ReasoningStep(BaseModel):
    """個別の推論ステップ"""
    step: int = Field(description="ステップ番号")
    subquestion: str = Field(description="このステップで答えるべき小問題")
    procedure: str = Field(description="実行する手順")
    result: str = Field(description="このステップの結果")

class StructuredResponse(BaseModel):
    """構造化された推論プロセスを持つ応答"""
    reasoning: list[ReasoningStep] = Field(
        description="段階的な推論ステップのリスト"
    )
    correct_answer: int = Field(
        description="最終的な正しい答え"
    )

client = instructor.from_openai(OpenAI())

response = client.chat.completions.create(
    model="gpt-4",
    response_model=StructuredResponse,
    messages=[
        {"role": "user", "content": "田中さんは5個のリンゴを持っています。3個食べて、その後友達から4個もらいました。今何個ありますか？"}
    ],
)

for step in response.reasoning:
    print(f"ステップ {step.step}: {step.subquestion}")
    print(f"  手順: {step.procedure}")
    print(f"  結果: {step.result}")

print(f"\n最終答え: {response.correct_answer}個")
```

### 出力例

```
ステップ 1: 最初の状態は？
  手順: 初期のリンゴの数を確認する
  結果: 5個

ステップ 2: 食べた後は？
  手順: 5個から3個を引く
  結果: 2個

ステップ 3: もらった後は？
  手順: 2個に4個を足す
  結果: 6個

最終答え: 6個
```

**参考資料**:
- [Structure The Reasoning - Instructor](https://python.useinstructor.com/prompting/thought_generation/chain_of_thought_zero_shot/tab_cot/)
- [Structured outputs with DeepSeek - Instructor](https://python.useinstructor.com/integrations/deepseek/)

---

## 3. Maybeパターン - 自然言語フォールバック

### 概要
構造化データを抽出できない場合に、自然言語メッセージでフォールバックする手法。LLMにエスケープハッチを提供することで、幻覚（ハルシネーション）を効果的に削減できます。

### 実装例

```python
from typing import Optional

class UserDetail(BaseModel):
    """ユーザー詳細情報"""
    name: str
    age: int
    email: str

class MaybeUser(BaseModel):
    """ユーザー情報の抽出結果（失敗時の自然言語メッセージ付き）"""
    result: Optional[UserDetail] = Field(
        default=None,
        description="抽出されたユーザー詳細情報"
    )
    error: bool = Field(
        default=False,
        description="抽出に失敗したかどうか"
    )
    message: Optional[str] = Field(
        default=None,
        description="エラーまたは説明のための自然言語メッセージ"
    )

client = instructor.from_openai(OpenAI())

# 成功ケース
response1 = client.chat.completions.create(
    model="gpt-4",
    response_model=MaybeUser,
    messages=[
        {"role": "user", "content": "田中太郎は35歳で、メールはtanaka@example.comです"}
    ],
)
# response1.result → UserDetail(name="田中太郎", age=35, email="tanaka@example.com")
# response1.error → False

# 失敗ケース（情報不足）
response2 = client.chat.completions.create(
    model="gpt-4",
    response_model=MaybeUser,
    messages=[
        {"role": "user", "content": "今日は良い天気ですね"}
    ],
)
# response2.result → None
# response2.error → True
# response2.message → "提供されたテキストにはユーザー情報が含まれていません"
```

### ユースケース
- 情報抽出が不可能な場合の安全な処理
- LLMが確信を持てない場合のフォールバック
- ハルシネーションの削減

**参考資料**:
- [Maybe Types and Optional Handling in Instructor](https://python.useinstructor.com/concepts/maybe/)

---

## 4. Self-Correction / Retry パターン

### 概要
バリデーションエラーが発生した際、エラー情報を**自然言語で**LLMにフィードバックし、自動的に再試行させる手法。

### 仕組み

1. LLMが構造化データを生成
2. Pydanticバリデーションでエラーが発生
3. Instructorがエラー内容を自然言語でLLMに送信
4. LLMがエラーを理解して修正した応答を生成
5. 最大リトライ回数まで繰り返す

### 実装例

```python
from pydantic import field_validator

class UserInfo(BaseModel):
    """ユーザー情報"""
    name: str = Field(description="ユーザーの名前")
    age: int = Field(description="年齢", ge=0, le=150)
    email: str = Field(description="メールアドレス")

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if '@' not in v:
            raise ValueError('有効なメールアドレスではありません。@記号を含む必要があります')
        return v

client = instructor.from_openai(OpenAI())

try:
    user = client.chat.completions.create(
        model="gpt-4",
        response_model=UserInfo,
        messages=[
            {"role": "user", "content": "太郎は25歳で、メールはtaro.example.comです"}
        ],
        max_retries=3  # 最大3回まで再試行
    )
except Exception as e:
    print(f"リトライ上限に達しました: {e}")
```

### 内部動作の例

**初回リクエスト**:
```
ユーザー入力: 太郎は25歳で、メールはtaro.example.comです
```

**LLMの初回応答**:
```json
{
  "name": "太郎",
  "age": 25,
  "email": "taro.example.com"  // エラー: @がない
}
```

**Instructorからのフィードバック（自然言語）**:
```
前回の応答にエラーがありました:
- フィールド 'email': 有効なメールアドレスではありません。@記号を含む必要があります

以下のスキーマで再度出力してください: {...}
```

**LLMの修正後の応答**:
```json
{
  "name": "太郎",
  "age": 25,
  "email": "taro@example.com"  // 修正された
}
```

**参考資料**:
- [Implementing Self-Correction with LLM Validator](https://python.useinstructor.com/examples/self_critique/)
- [Python Retry Logic with Tenacity and Instructor](https://python.useinstructor.com/concepts/retrying/)
- [Validation in Instructor](https://python.useinstructor.com/concepts/validation/)

---

## 5. Plan and Solve パターン

### 概要
Zero-Shot Chain of Thoughtを改善したアプローチ。より詳細な指示をプロンプトに追加し、**計画→実行**の2段階プロセスを実現します。

### 実装イメージ

```python
class PlanStep(BaseModel):
    """計画の1ステップ"""
    step_number: int
    description: str
    expected_outcome: str

class Plan(BaseModel):
    """実行計画"""
    goal: str = Field(description="達成したい目標")
    steps: list[PlanStep] = Field(description="実行ステップのリスト")

class Solution(BaseModel):
    """計画と解決策"""
    plan: Plan = Field(description="問題解決のための計画")
    final_answer: str = Field(description="最終的な答え")

client = instructor.from_openai(OpenAI())

response = client.chat.completions.create(
    model="gpt-4",
    response_model=Solution,
    messages=[
        {
            "role": "system",
            "content": "まず計画を立て、その後段階的に問題を解決してください"
        },
        {
            "role": "user",
            "content": "3日間の東京旅行の予算を計算してください。宿泊費は1泊1万円、食費は1日5千円、交通費は1日2千円です"
        }
    ],
)

print("計画:")
for step in response.plan.steps:
    print(f"{step.step_number}. {step.description} → {step.expected_outcome}")

print(f"\n最終答え: {response.final_answer}")
```

**参考資料**:
- [Ditch Vanilla Chain Of Thought - Instructor](https://python.useinstructor.com/prompting/decomposition/plan_and_solve/)

---

## 実装可能性の検証

### Q1: Instructorフレームワークと合わせて2段階応答は実装可能か？

**✅ 完全に可能です**

以下の複数の方法で実装できます：

1. **単一モデル内での2段階**: `chain_of_thought` + `final_answer` フィールドを持つモデル
2. **構造化された推論**: `reasoning: list[Step]` + `answer` の形式
3. **Maybe型による柔軟な応答**: 構造化データまたは自然言語メッセージ
4. **Self-Correctionループ**: エラー時の自然言語フィードバック
5. **Plan and Solve**: 計画フェーズと実行フェーズの明示的な分離

### Q2: Instructorのドキュメントに記載があるか？

**✅ 公式ドキュメントに詳細な記載があります**

以下のトピックで複数のページにわたって説明されています：

- **Chain of Thought**: 基本的なCoTパターンの実装方法
- **Tab-CoT**: 構造化された推論テーブル
- **Maybe Types**: 自然言語フォールバック
- **Retry/Self-Correction**: バリデーションエラー時の自己修正
- **Advanced Prompting**: Plan and Solveなどの高度なパターン

---

## 推奨実装パターン: 小規模LLM向け

小規模LLMで最も精度が高いとされる「自然言語→自然言語」の特性を活かしつつ、構造化出力を得る推奨パターン：

### パターンA: Chain of Thought with Exclusion

```python
class SmallLLMResponse(BaseModel):
    """小規模LLM向けの応答モデル"""

    # 自然言語での推論プロセス（LLMが得意）
    thinking: str = Field(
        description="自然な日本語で考えるプロセス。専門用語を使わず、人間のように考えてください",
        exclude=True  # 内部処理のみで使用、外部には出力しない
    )

    # 最終的な構造化データ
    answer: str = Field(description="最終的な答え")
    confidence: float = Field(description="確信度 (0.0-1.0)", ge=0.0, le=1.0)
```

### パターンB: Maybe with Natural Language First

```python
class FlexibleResponse(BaseModel):
    """柔軟な応答モデル - 自然言語優先"""

    # まず自然言語で応答
    natural_response: str = Field(
        description="自然な日本語での完全な回答"
    )

    # 可能であれば構造化データも提供
    structured_data: Optional[dict] = Field(
        default=None,
        description="自然言語応答から抽出できる場合のみ、構造化データを提供"
    )

    extraction_confidence: float = Field(
        description="構造化データの抽出確信度",
        ge=0.0,
        le=1.0
    )
```

### パターンC: Two-Stage Explicit (2回呼び出し)

本当に2段階に分けたい場合は、2回のLLM呼び出しを行う方法もあります：

```python
# ステップ1: 自然言語で応答を生成
class NaturalResponse(BaseModel):
    answer: str = Field(description="自然な日本語での回答")

natural_resp = client.chat.completions.create(
    model="llama3.1:8b",
    response_model=NaturalResponse,
    messages=[{"role": "user", "content": user_question}],
)

# ステップ2: 自然言語応答を構造化データに変換
class StructuredData(BaseModel):
    name: str
    age: int
    occupation: str

structured_resp = client.chat.completions.create(
    model="llama3.1:8b",
    response_model=StructuredData,
    messages=[
        {
            "role": "user",
            "content": f"以下のテキストから情報を抽出してください:\n{natural_resp.answer}"
        }
    ],
)
```

**メリット**:
- 第1段階で自然言語生成の精度を最大化
- 第2段階で構造化に専念
- 各段階で異なるモデルやプロンプト戦略を使用可能

**デメリット**:
- 2回の呼び出しでコストと時間が増加
- 情報の損失リスク

---

## まとめ

### 調査結果

1. **✅ Instructorは2段階応答パターンと完全に互換性あり**
   - Chain of Thought、Tab-CoT、Maybe、Self-Correction、Plan and Solveなど複数の手法を公式サポート

2. **✅ 公式ドキュメントに詳細な記載あり**
   - 実装例、ベストプラクティス、パフォーマンス改善データが豊富

3. **✅ 小規模LLMでの精度向上が実証済み**
   - GSM8kで60%の精度向上
   - 自然言語での推論プロセスが有効

### 推奨アプローチ

**小規模LLM（Ollama + llama3.1:8bなど）の場合**:
- `chain_of_thought`フィールドを使った単一モデルアプローチ
- フィールド名と説明文を日本語で詳細に記述
- `exclude=True`で推論プロセスを内部処理のみに使用
- `max_retries`でSelf-Correctionを活用

**より高度な制御が必要な場合**:
- 2回呼び出しパターン（自然言語生成→構造化変換）
- Tab-CoTで推論プロセスを構造化して追跡
- Maybeパターンで抽出失敗時の安全な処理

---

## 参考資料

### 主要ドキュメント
- [Structure The Reasoning - Instructor](https://python.useinstructor.com/prompting/thought_generation/chain_of_thought_zero_shot/tab_cot/)
- [Text Classification with OpenAI and Pydantic](https://python.useinstructor.com/examples/classification/)
- [Maybe Types and Optional Handling](https://python.useinstructor.com/concepts/maybe/)
- [Implementing Self-Correction with LLM Validator](https://python.useinstructor.com/examples/self_critique/)
- [Python Retry Logic with Tenacity and Instructor](https://python.useinstructor.com/concepts/retrying/)
- [Ditch Vanilla Chain Of Thought](https://python.useinstructor.com/prompting/decomposition/plan_and_solve/)
- [Advanced Prompting Techniques Guide](https://python.useinstructor.com/prompting/)

### 関連記事
- [Bad Schemas could break your LLM Structured Outputs](https://python.useinstructor.com/blog/2024/09/26/bad-schemas-could-break-your-llm-structured-outputs/)
- [Good LLM Validation is Just Good Validation](https://python.useinstructor.com/blog/2023/10/23/good-llm-validation-is-just-good-validation/)
- [Bridging Language Models with Python with Instructor, Pydantic, and OpenAI's function calls (Medium)](https://medium.com/@jxnlco/bridging-language-model-with-python-with-instructor-pydantic-and-openais-function-calling-f32fb1cdb401)

---

**作成日**: 2025-12-19
**調査者**: Claude (Sonnet 4.5)
**目的**: LLMの2段階応答パターンとInstructorフレームワークの統合可能性調査
