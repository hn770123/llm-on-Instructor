# Instructor フレームワーク デモアプリケーション

このプロジェクトは、**Instructor フレームワーク**を使用して、LLM（Large Language Model）から**構造化された出力**を取得するWebアプリケーションのサンプルです。

## 📋 概要

- **フレームワーク**: Instructor（Python）
- **LLM**: Ollama (llama3.1:8b)
- **バックエンド**: FastAPI
- **フロントエンド**: HTML + JavaScript
- **機能**:
  - プロンプト入力
  - 複数の出力フォーマットから選択可能
  - カスタムスキーマの定義
  - 構造化されたJSON出力

## 🚀 特徴

- **型安全な出力**: Pydanticモデルを使用した厳密な型定義
- **複数のフォーマット対応**:
  - ユーザー情報（名前、年齢、メール、職業）
  - 製品情報（製品名、価格、カテゴリー、説明、在庫）
  - 記事情報（タイトル、著者、要約、タグ、公開日）
  - タスクリスト（プロジェクト名、タスク詳細）
  - カスタムフォーマット（自由なスキーマ定義）
- **Webインターフェース**: 使いやすいUIで操作可能
- **詳細なコメント**: 学習用に日本語コメントを豊富に記載

## 📁 プロジェクト構成

```
llm-on-Instructor/
├── backend/
│   └── app.py                          # FastAPI バックエンドアプリケーション
├── frontend/
│   └── index.html                      # Webフロントエンドインターフェース
├── docs/
│   ├── instructor-flow-diagram.md      # Instructorフローダイアグラム
│   └── two-stage-response-patterns.md  # 2段階応答パターン調査結果
├── examples/
│   └── two_stage_patterns.py           # 2段階応答パターンの実装例
├── requirements.txt                    # Python依存関係
├── README.md                          # このファイル
└── research-Instructor.md             # Instructorフレームワークの調査資料
```

## 🔧 セットアップ

### 前提条件

1. **Python 3.8以上**がインストールされていること
2. **Ollama**がインストールされ、実行中であること
3. **llama3.1:8bモデル**がOllamaにダウンロードされていること

### Ollamaのセットアップ

```bash
# Ollamaのインストール（macOS/Linux）
curl -fsSL https://ollama.com/install.sh | sh

# llama3.1:8bモデルのダウンロード
ollama pull llama3.1:8b

# Ollamaサーバーの起動（通常は自動で起動）
ollama serve
```

### Pythonパッケージのインストール

```bash
# リポジトリをクローン
git clone <repository-url>
cd llm-on-Instructor

# 依存関係をインストール
pip install -r requirements.txt
```

## 🎯 使い方

### 1. バックエンドサーバーの起動

```bash
# backendディレクトリに移動
cd backend

# FastAPIサーバーを起動
python app.py
```

サーバーは `http://localhost:8000` で起動します。

### 2. フロントエンドを開く

ブラウザで `frontend/index.html` を直接開くか、以下のコマンドでローカルサーバーを起動します：

```bash
# PythonのシンプルなHTTPサーバーを使用
cd frontend
python -m http.server 8080
```

その後、ブラウザで `http://localhost:8080` にアクセスします。

### 3. アプリケーションの使用

1. **プロンプトを入力**: テキストエリアにLLMに送るプロンプトを入力
2. **フォーマットを選択**: ドロップダウンから出力フォーマットを選択
3. **生成ボタンをクリック**: LLMが構造化された出力を生成
4. **結果を確認**: 右側のパネルにJSON形式で結果が表示されます

### 例文の使用

各フォーマットに対応した例文が表示されます。例文をクリックすると、プロンプト入力エリアに自動入力されます。

## 📊 出力フォーマット

### 1. ユーザー情報 (`user`)

```json
{
  "name": "田中太郎",
  "age": 35,
  "email": "tanaka@example.com",
  "occupation": "エンジニア"
}
```

### 2. 製品情報 (`product`)

```json
{
  "product_name": "iPhone 15",
  "price": 149800.0,
  "category": "スマートフォン",
  "description": "最新のA17チップを搭載",
  "in_stock": true
}
```

### 3. 記事情報 (`article`)

```json
{
  "title": "AIの未来",
  "author": "山田花子",
  "summary": "人工知能の発展について論じています",
  "tags": ["AI", "機械学習", "未来"],
  "publication_date": "2024-01-15"
}
```

### 4. タスクリスト (`tasklist`)

```json
{
  "project_name": "Webサイトリニューアル",
  "tasks": [
    {
      "task_name": "デザイン作成",
      "priority": "高",
      "deadline": "2024-02-01",
      "completed": false
    }
  ],
  "total_tasks": 3
}
```

### 5. カスタムフォーマット (`custom`)

自由にスキーマを定義できます。JSON形式でフィールド定義を入力してください。

## 🛠️ API エンドポイント

### GET `/`
- APIの基本情報を返します

### GET `/health`
- ヘルスチェック - Ollamaの接続状態を確認

### GET `/formats`
- 利用可能な出力フォーマットの一覧を返します

### POST `/generate`
- LLMから構造化出力を生成

**リクエストボディ:**
```json
{
  "prompt": "プロンプト文字列",
  "format_type": "user|product|article|tasklist|custom",
  "custom_schema": "カスタムスキーマ（オプション）"
}
```

**レスポンス:**
```json
{
  "success": true,
  "data": { ... },
  "format_type": "user",
  "message": "構造化出力の生成に成功しました"
}
```

## 📚 技術スタック

### バックエンド
- **FastAPI**: 高速で使いやすいWebフレームワーク
- **Instructor**: LLMから構造化出力を取得するフレームワーク
- **Pydantic**: データ検証とスキーマ定義
- **Ollama**: ローカルLLMサーバー
- **Uvicorn**: ASGIサーバー

### フロントエンド
- **HTML5**: マークアップ
- **CSS3**: スタイリング（グラデーション、レスポンシブデザイン）
- **JavaScript (Vanilla)**: UIロジックとAPI通信

## 🎓 学習ポイント

このプロジェクトから学べること：

1. **Instructorフレームワークの基本**: LLMから構造化出力を取得する方法
2. **Pydanticモデル**: 型安全なデータ定義
3. **FastAPI**: RESTful APIの構築
4. **Ollama統合**: ローカルLLMの活用
5. **フロントエンド/バックエンド通信**: Fetch APIの使用
6. **エラーハンドリング**: 適切なエラー処理とユーザーフィードバック
7. **2段階応答パターン**: Chain of Thought、Maybe、Self-Correctionなどの高度な手法

## 🔍 コードの解説

### バックエンド (`backend/app.py`)

- **Pydanticモデル定義**: 各出力フォーマットに対応したモデルを定義
- **Instructorクライアント**: Ollamaクライアントにパッチを適用
- **エンドポイント実装**: FastAPIのルートとリクエスト処理
- **エラーハンドリング**: 適切な例外処理とHTTPレスポンス

### フロントエンド (`frontend/index.html`)

- **レスポンシブUI**: モバイルフレンドリーなデザイン
- **動的な例文表示**: フォーマットに応じた例文の切り替え
- **API通信**: Fetch APIを使用した非同期通信
- **ステータス表示**: ローディング、成功、エラーの視覚的フィードバック

## ⚠️ 注意事項

- **本番環境での使用**: CORS設定、セキュリティ、エラーハンドリングを強化してください
- **Ollamaの起動**: バックエンド起動前にOllamaサーバーが実行中であることを確認
- **モデルのダウンロード**: llama3.1:8bモデルが大きいため、初回ダウンロードに時間がかかる場合があります
- **リソース使用**: LLMの実行にはCPU/GPUリソースが必要です

## 🤝 トラブルシューティング

### Ollamaに接続できない

```bash
# Ollamaが実行中か確認
ps aux | grep ollama

# Ollamaを再起動
ollama serve
```

### モジュールが見つからないエラー

```bash
# 依存関係を再インストール
pip install -r requirements.txt --force-reinstall
```

### CORS エラー

フロントエンドをローカルHTTPサーバー経由で開くか、バックエンドのCORS設定を確認してください。

## 📝 ライセンス

このプロジェクトはサンプル/学習目的で作成されています。

## 📚 ドキュメント

このプロジェクトには以下のドキュメントが含まれています：

### [2段階応答パターン調査](docs/two-stage-response-patterns.md)
LLMの応答を自然言語→構造化データの2段階にする手法の詳細な調査結果。以下のパターンを網羅：
- Chain of Thought (CoT)
- Tabular Chain of Thought (Tab-CoT)
- Maybeパターン（自然言語フォールバック）
- Self-Correction / Retry
- Plan and Solve
- 小規模LLM向けの推奨実装パターン

### [Instructorフローダイアグラム](docs/instructor-flow-diagram.md)
Instructorフレームワークの動作フロー、バリデーション、リトライメカニズムの詳細な図解。

### [実装例](examples/two_stage_patterns.py)
8つの異なる2段階応答パターンの実装例。すぐに試せるサンプルコード付き。

## 🙏 参考資料

- [Instructor Framework 公式ドキュメント](https://python.useinstructor.com/)
- [FastAPI 公式ドキュメント](https://fastapi.tiangolo.com/)
- [Ollama 公式サイト](https://ollama.com/)
- [Pydantic ドキュメント](https://docs.pydantic.dev/)

---

**作成者**: Claude
**作成日**: 2024
**目的**: Instructorフレームワークの学習とデモンストレーション
