import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # 現在は可視デバイスが設定されていないため、TensorFlowはGPU 0を使用
    # メモリの増加を許可
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ {len(gpus)} 個のGPUが認識されています:")
    print(gpus)
  except RuntimeError as e:
    # エラーが発生した場合の処理
    print(e)
else:
  print("⚠️ GPUが認識されていません。")