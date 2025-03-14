from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from typing import List, Optional
from ..models.recommender.graph_rec import GraphRecommender
from ..models.gnn.gat import GATRecommender

app = FastAPI(title="Graph-based Recommender System API")

# 加载模型（在实际应用中，这些变量应该通过配置文件或环境变量设置）
MODEL_PATH = "best_model.pth"
NUM_USERS = 1000  # 示例值，实际应用中需要根据实际数据设置
NUM_ITEMS = 5000  # 示例值，实际应用中需要根据实际数据设置
HIDDEN_CHANNELS = 64
MODEL_TYPE = "gat"  # 或 "gcn"

# 创建模型实例
if MODEL_TYPE == "gat":
    model = GATRecommender(NUM_USERS, NUM_ITEMS, HIDDEN_CHANNELS)
else:
    model = GraphRecommender(NUM_USERS, NUM_ITEMS, HIDDEN_CHANNELS)

# 加载预训练的模型参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

class RecommendationRequest(BaseModel):
    user_id: int
    n_recommendations: Optional[int] = 10

class ItemScoreResponse(BaseModel):
    item_id: int
    score: float

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[ItemScoreResponse]

@app.get("/")
async def root():
    return {"message": "Graph-based Recommender System API"}

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    try:
        # 检查用户ID是否有效
        if request.user_id >= NUM_USERS:
            raise HTTPException(status_code=400, detail="Invalid user ID")

        # 准备模型输入
        user_tensor = torch.tensor([request.user_id], device=device)
        
        with torch.no_grad():
            # 获取推荐
            top_scores, top_indices = model.recommend(
                user_tensor,
                model.edge_index,  # 这里假设模型中存储了图的边信息
                top_k=request.n_recommendations
            )

        # 准备响应
        recommendations = []
        for item_id, score in zip(top_indices[0].cpu().numpy(), top_scores[0].cpu().numpy()):
            recommendations.append(ItemScoreResponse(
                item_id=int(item_id),
                score=float(score)
            ))

        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def get_model_info():
    return {
        "model_type": MODEL_TYPE,
        "num_users": NUM_USERS,
        "num_items": NUM_ITEMS,
        "hidden_channels": HIDDEN_CHANNELS,
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)