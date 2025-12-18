"""
Instructor フレームワークを使用した構造化出力サンプルアプリケーション

このアプリケーションは、Ollama (llama3.1:8b) とInstructorフレームワークを使って、
LLMから指定した型で構造化された応答を受け取るWebアプリケーションです。
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import instructor
import ollama
import json

# FastAPIアプリケーションのインスタンスを作成
app = FastAPI(
    title="Instructor Framework Demo",
    description="Instructorフレームワークを使用したLLM構造化出力デモ",
    version="1.0.0"
)

# CORS設定 - フロントエンドからのリクエストを許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では特定のオリジンのみ許可すべき
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# 出力フォーマット用のPydanticモデル定義
# =============================================================================

# フォーマット1: ユーザー情報
class UserInfo(BaseModel):
    """ユーザー情報を表す構造化モデル"""
    name: str = Field(description="ユーザーの名前")
    age: int = Field(description="ユーザーの年齢", ge=0, le=150)
    email: Optional[str] = Field(default=None, description="メールアドレス")
    occupation: Optional[str] = Field(default=None, description="職業")

# フォーマット2: 製品情報
class ProductInfo(BaseModel):
    """製品情報を表す構造化モデル"""
    product_name: str = Field(description="製品名")
    price: float = Field(description="価格", ge=0)
    category: str = Field(description="カテゴリー")
    description: str = Field(description="製品の説明")
    in_stock: bool = Field(description="在庫の有無")

# フォーマット3: 記事情報
class ArticleInfo(BaseModel):
    """記事情報を表す構造化モデル"""
    title: str = Field(description="記事のタイトル")
    author: str = Field(description="著者名")
    summary: str = Field(description="記事の要約", max_length=500)
    tags: List[str] = Field(description="タグリスト")
    publication_date: Optional[str] = Field(default=None, description="公開日")

# フォーマット4: タスクリスト
class Task(BaseModel):
    """個別のタスク"""
    task_name: str = Field(description="タスク名")
    priority: str = Field(description="優先度 (高/中/低)")
    deadline: Optional[str] = Field(default=None, description="締切日")
    completed: bool = Field(default=False, description="完了状態")

class TaskList(BaseModel):
    """タスクリストを表す構造化モデル"""
    project_name: str = Field(description="プロジェクト名")
    tasks: List[Task] = Field(description="タスクのリスト")
    total_tasks: int = Field(description="総タスク数")

# フォーマット5: カスタムフォーマット（動的生成）
class CustomFormat(BaseModel):
    """カスタムフォーマット用の基底モデル"""
    data: Dict[str, Any] = Field(description="カスタムデータ")

# =============================================================================
# リクエスト/レスポンスモデル
# =============================================================================

class InstructorRequest(BaseModel):
    """APIリクエストのモデル"""
    prompt: str = Field(description="LLMに送るプロンプト")
    format_type: str = Field(
        description="出力フォーマットの種類",
        pattern="^(user|product|article|tasklist|custom)$"
    )
    custom_schema: Optional[str] = Field(
        default=None,
        description="カスタムフォーマット使用時のスキーマ定義（JSON文字列）"
    )

class InstructorResponse(BaseModel):
    """APIレスポンスのモデル"""
    success: bool = Field(description="処理成功フラグ")
    data: Any = Field(description="構造化された出力データ")
    format_type: str = Field(description="使用されたフォーマットタイプ")
    message: Optional[str] = Field(default=None, description="メッセージ")

# =============================================================================
# Instructorクライアントの初期化
# =============================================================================

def get_instructor_client():
    """
    Instructorクライアントを初期化して返す

    OllamaクライアントにInstructorをパッチして、
    構造化出力機能を追加します。
    """
    # Ollamaクライアントを作成
    ollama_client = ollama.Client()

    # InstructorでOllamaクライアントをパッチ
    # これにより、response_modelパラメータが使用可能になる
    client = instructor.from_openai(
        ollama_client,
        mode=instructor.Mode.JSON  # JSON形式で構造化出力を取得
    )

    return client

# =============================================================================
# フォーマットタイプとPydanticモデルのマッピング
# =============================================================================

FORMAT_MODELS = {
    "user": UserInfo,
    "product": ProductInfo,
    "article": ArticleInfo,
    "tasklist": TaskList,
    "custom": CustomFormat
}

# =============================================================================
# エンドポイント
# =============================================================================

@app.get("/")
async def root():
    """ルートエンドポイント - APIの情報を返す"""
    return {
        "message": "Instructor Framework Demo API",
        "version": "1.0.0",
        "endpoints": {
            "/generate": "POST - LLMから構造化出力を生成",
            "/formats": "GET - 利用可能な出力フォーマット一覧",
            "/health": "GET - ヘルスチェック"
        }
    }

@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    try:
        # Ollamaが起動しているか確認
        ollama_client = ollama.Client()
        # 利用可能なモデル一覧を取得
        models = ollama_client.list()
        return {
            "status": "healthy",
            "ollama_available": True,
            "models_count": len(models.get("models", []))
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "ollama_available": False,
            "error": str(e)
        }

@app.get("/formats")
async def get_formats():
    """利用可能な出力フォーマットの一覧を返す"""
    return {
        "formats": {
            "user": {
                "name": "ユーザー情報",
                "description": "名前、年齢、メール、職業などのユーザー情報",
                "fields": ["name", "age", "email", "occupation"],
                "example_prompt": "田中太郎は35歳のエンジニアで、メールアドレスはtanaka@example.comです"
            },
            "product": {
                "name": "製品情報",
                "description": "製品名、価格、カテゴリー、説明、在庫状況",
                "fields": ["product_name", "price", "category", "description", "in_stock"],
                "example_prompt": "iPhone 15は149,800円のスマートフォンで、最新のA17チップを搭載しています。在庫あり"
            },
            "article": {
                "name": "記事情報",
                "description": "タイトル、著者、要約、タグ、公開日",
                "fields": ["title", "author", "summary", "tags", "publication_date"],
                "example_prompt": "山田花子が書いた「AIの未来」という記事は、人工知能の発展について論じています。タグ: AI, 機械学習, 未来"
            },
            "tasklist": {
                "name": "タスクリスト",
                "description": "プロジェクト名と複数のタスク情報",
                "fields": ["project_name", "tasks"],
                "example_prompt": "Webサイトリニューアルプロジェクトで、デザイン作成(高優先度)、コーディング(中優先度)、テスト(低優先度)の3つのタスクがあります"
            },
            "custom": {
                "name": "カスタムフォーマット",
                "description": "ユーザー定義の自由な構造",
                "fields": ["dynamic"],
                "example_prompt": "カスタムスキーマを定義してください"
            }
        }
    }

@app.post("/generate", response_model=InstructorResponse)
async def generate_structured_output(request: InstructorRequest):
    """
    LLMから構造化された出力を生成するメインエンドポイント

    Args:
        request: プロンプト、フォーマットタイプ、カスタムスキーマを含むリクエスト

    Returns:
        構造化された出力データを含むレスポンス
    """
    try:
        # フォーマットタイプの検証
        if request.format_type not in FORMAT_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"無効なフォーマットタイプです: {request.format_type}"
            )

        # 使用するPydanticモデルを取得
        response_model = FORMAT_MODELS[request.format_type]

        # カスタムフォーマットの場合は、スキーマを動的に生成
        if request.format_type == "custom" and request.custom_schema:
            try:
                # カスタムスキーマをパース
                schema_dict = json.loads(request.custom_schema)

                # 動的にPydanticモデルを生成
                # (実際の実装では、より詳細なバリデーションが必要)
                response_model = type(
                    "DynamicModel",
                    (BaseModel,),
                    {"__annotations__": schema_dict}
                )
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="カスタムスキーマのJSON形式が無効です"
                )

        # Instructorクライアントを初期化
        # 注: この実装は簡略化されています。実際のOllama + Instructor統合は
        # より複雑な設定が必要な場合があります

        # Ollamaクライアントを使用してLLMに問い合わせ
        ollama_client = ollama.Client()

        # プロンプトにフォーマット指示を追加
        enhanced_prompt = f"""
{request.prompt}

以下のJSON形式で出力してください:
{response_model.model_json_schema()}
"""

        # Ollamaを使用してLLMから応答を取得
        response = ollama_client.chat(
            model="llama3.1:8b",
            messages=[
                {
                    "role": "system",
                    "content": "あなたは指定されたJSON形式で正確に応答するアシスタントです。"
                },
                {
                    "role": "user",
                    "content": enhanced_prompt
                }
            ],
            format="json"  # JSON形式での出力を強制
        )

        # LLMの応答をパース
        llm_output = response["message"]["content"]

        # JSON文字列をPydanticモデルに変換
        structured_data = response_model.model_validate_json(llm_output)

        # レスポンスを返す
        return InstructorResponse(
            success=True,
            data=structured_data.model_dump(),
            format_type=request.format_type,
            message="構造化出力の生成に成功しました"
        )

    except HTTPException:
        # HTTPExceptionはそのまま再発生
        raise
    except Exception as e:
        # その他のエラーをキャッチして適切なレスポンスを返す
        raise HTTPException(
            status_code=500,
            detail=f"構造化出力の生成中にエラーが発生しました: {str(e)}"
        )

# =============================================================================
# アプリケーションのエントリーポイント
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    # Uvicornサーバーを起動
    # ホスト: 0.0.0.0 (すべてのインターフェースでリッスン)
    # ポート: 8000
    # リロード: True (開発時のみ、ファイル変更時に自動リロード)
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
