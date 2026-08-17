"""訓練 log 的共用工具：把 train.py 印出來的東西同時寫進
model/<version>/result/result.log，開頭記錄 dataset/ 的重要超參數跟這次
訓練用的超參數，方便之後回頭比較不同版本、不同資料設定跑出來的結果。
"""

from config.dataset import _PATCH_PARAMS, result


def open_log(version, hparams):
    """開一個新的 result.log，寫入 dataset 設定跟這次訓練的超參數。

    version：模型版本字串，決定寫到 model/<version>/result/result.log
    （沿用 config.dataset.result() 的路徑規則）。
    hparams：dict，這次訓練用的超參數（EPOCHS、LR……），攤平寫進 header。

    回傳 log(msg)：呼叫時同時 print 到 stdout、寫進這個 log 檔（覆蓋舊檔
    重開，逐行 append）並立刻 flush，訓練中途中斷也讀得到目前為止的紀錄。
    """
    f = open(result(version, "result.log"), "w")

    def _section(title, d):
        f.write(f"[{title}]\n")
        for k, v in d.items():
            f.write(f"  {k} = {v}\n")
        f.write("\n")

    _section("dataset", _PATCH_PARAMS)
    _section("hparams", hparams)
    f.flush()

    def log(msg):
        print(msg)
        f.write(msg + "\n")
        f.flush()

    return log
