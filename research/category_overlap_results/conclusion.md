# POI 類別重疊量化診斷

## 結論

餐飲類面積過大的主因不是繪圖遮蓋，而是 `dominant = argmax(category proportion)` 將原本的組成資料壓成單一硬標籤後，放大了原始資料的不平衡；二維 latent 壓縮會再增加一些混合，但屬於次要原因。

原圖的繪製順序也是先畫餐飲、再畫其他類別，因此其他類別實際上位於餐飲點的上層；單純調整圖層順序不能解決量化上的重疊。

## 主要證據

| 診斷 | 結果 | 解讀 |
|---|---:|---|
| 餐飲占全部 POI count | 50.32% | 原始資料本身明顯不平衡 |
| 餐飲占 dominant patch | 84.67%（1044/1233） | 硬 argmax 將餐飲占比放大 1.68 倍 |
| 從未成為 dominant 的類別 | 2 類 | 醫療、運動休閒無法出現在 dominant 圖中 |
| top-1 與 top-2 差距小於 0.15 | 25.30% | 四分之一網格本來就是混合且標籤不穩定 |
| 非餐飲 dominant 網格中的平均餐飲比例 | 27.45% | 餐飲成分確實廣泛存在於其他類網格 |
| 10D 原始 composition 的 15-NN macro-F1 | 0.443 | 未壓縮前也無法形成完整純類別群 |
| 高信心網格的原始 composition macro-F1 | 0.749 | 排除混合網格後，可分性大幅提高 |
| 最佳舊模型 2D latent macro-F1 | 0.376（DAE） | 二維壓縮再損失約 0.067 |
| DDAE+GAT 融合 latent macro-F1 | 0.368 | GAT 未消除餐飲重疊 |
| GAT count / OD 分支 macro-F1 | 0.264 / 0.251 | 兩分支單獨皆弱，融合後才恢復部分資訊 |
| latent 中非餐飲點的餐飲鄰居比例 | 48.75%–60.88% | 所有二維 latent 都有明顯餐飲鄰域污染 |

所有二維 latent 的 silhouette score 都小於 0，表示以 dominant 類別衡量時，各類並不是緊密且互相分離的群集。

## 建議改善順序

1. 不再把每個網格視為單一純類別。保留完整類別比例，並將 `top1 - top2 < 0.15` 的網格標為「混合」，或用比例／entropy 直接呈現。
2. 若要讓 latent 具有類別組成可分性，在既有 Poisson reconstruction 與 FSCE 之外，加一個預測完整類別比例的輔助 head，使用 class-balanced soft cross-entropy 或 Jensen-Shannon loss；不要只監督 dominant hard label。
3. 將總 POI 量與類別組成分開建模，避免餐飲的總量同時主導 reconstruction 與 latent 距離。
4. 改善實驗需同時報告 clean validation Poisson deviance、composition divergence、macro-F1、低信心網格比例，以及非餐飲點的餐飲鄰居比例，避免用 accuracy 被 84.67% 的餐飲多數類掩蓋。

完整數值位於 `category_data_summary.csv`、`representation_metrics.csv`、`gat_confusion_matrix.csv` 與 `summary.json`；診斷程式為 `../category_overlap_experiment.py`。
