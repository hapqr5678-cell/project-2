"""訓練 log 的共用工具：把 train.py 印出來的東西同時寫進
model/<version>/result/result.log，開頭記錄 dataset/ 的重要超參數跟這次
訓練用的超參數，方便之後回頭比較不同版本、不同資料設定跑出來的結果。
"""

from config.dataset import _PATCH_PARAMS, result


def open_log(version, hparams):
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
