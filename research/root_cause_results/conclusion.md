# POI latent 重疊根因診斷

## 核心結論

問題不是「資料完全沒有類別結構」，也不是「任何二維表示都不可能」。真正衝突是：目前同一個二維 latent 同時被要求保存 POI 總量以重建 count，又要呈現類別組成；Poisson reconstruction 明顯使它優先保存總量，而不是 composition 幾何。

## 證據

### 1. 二維 composition 幾何其實可行

只用 training split fitting、validation transform 的 composition UMAP-2D：

- held-out macro-F1：`0.655`
- confident macro-F1：`1.000`
- 非餐飲 validation 點的餐飲鄰居比例：`0.292`
- composition R2：`0.783`
- log total count R2：`-0.020`

因此二維空間可以表達 composition，但幾乎不保存總量。

### 2. Original GAT 選擇了相反的資訊

Original GAT fused latent：

- held-out macro-F1：`0.377`
- 非餐飲 validation 點的餐飲鄰居比例：`0.579`
- composition R2：`0.488`
- log total count R2：`0.848`

它很適合保存總量與進行 Poisson count reconstruction，但犧牲了 composition 的鄰域結構。這正是 dominant-category plot 重疊的直接原因。

### 3. 非餐飲結構沒有消失在原始資料裡

排除 Dining 後，以其他九類的 conditional composition 判定 secondary category：

- 原始 9D non-Dining composition macro-F1：`0.695`
- Original GAT count branch：`0.416`
- Original GAT fused：`0.378`
- Original GAT OD branch：`0.236`

其他類別的結構存在，但二維 count reconstruction latent 沒有保留下來；固定 OD fusion 還會進一步降低 secondary-category 資訊。

### 4. OD graph 有訊號，但不是強烈的 patch-category graph

- trip-weighted same-category edge fraction：`0.496`
- 依邊際類別分布計算的期望值：`0.350`
- category assortativity lift：`1.415`
- connected-patch composition similarity：`0.851`
- random destination baseline：`0.795`
- patch similarity lift：`1.069`

OD graph 確實有類別訊號，所以不應直接丟棄；但在 patch composition 層級只比隨機高約 6.9%，不足以單獨建立乾淨類別群集。它較適合服務流動與重建表徵，而不是直接主導 composition 的二維位置。

### 5. Dominant label 仍有先天限制

- Dining：50.3% POI count，卻是 84.7% patch 的 dominant label。
- 25.3% patch 的 top-1/top-2 margin 小於 0.15。
- Health、Sports 從未成為 dominant；Sports 甚至從未成為排除 Dining 後的 secondary dominant。

因此任何方法都不可能讓 dominant 圖出現十個大小相近且完整的群集。

## 下一個最小風險實驗

不再向同一個二維 latent 添加分類或 pair loss。改成共享 encoder hidden representation、兩個用途分離的 latent heads：

```text
count + OD -> shared hidden representation
                  |--- h_recon (8D) -> Poisson decoder
                  |
                  +--- z_comp (2D)  -> training-only composition fuzzy graph
```

- `h_recon` 保存總量、OD 與重建資訊。
- `z_comp` 只保存 composition 鄰域，使用和成功 UMAP 對照一致的 training-only composition cosine kNN fuzzy graph。
- 不使用 dominant hard labels、不使用 composition classifier、不使用 global balanced random pairs。
- 第一階段先不讓 OD 進入 `z_comp`，只驗證 count composition head 能否接近 UMAP-2D 的 `0.655` macro-F1。
- 通過後才測試 gated OD contribution；不要再直接固定 `alpha_od=0.3` 混入 composition latent。

這個設計不是再猜一種 loss，而是由診斷顯示的「總量與 composition 需要分開表示」直接推導。
