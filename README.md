# FP8 혼합 포맷 depthwise 추론 실험

MyChip_FP_MAC 칩 연산으로 MobileNetV1(CIFAR-10)의 depthwise 레이어를 추론하는 실험.
모든 명령은 이 폴더(`inference`)에서 실행한다.

## 폴더
```
fp8/chip_ref.py     칩 스칼라 기준 모델 (mpw_test/smoke_gen/model.py 복사본, 수정 금지)
fp8/chip_torch.py   칩 비트 정확 모델 (PyTorch 텐서, ReLU on/off)
fp8/quant.py        FP32 <-> E4M3/E5M2 코드 변환 (스케일 없음)
models/mobilenetv1.py
cifar.py            CIFAR-10 로더, 평가 함수
tests/test_step0.py 단계 0 검증
train.py            단계 1-1 FP32 학습
fold_bn.py          단계 1-2 BN 합치기 + 확인
analyze.py          단계 2 분포 분석
fp8/dwconv.py       depthwise 한 레이어를 칩 연산으로 수행
fp8/net.py          depthwise 13개만 칩 연산으로 바꾼 모델
tests/test_step3.py dwconv 검증
run_uniform.py      단계 3 동일 포맷 8조합 평가
fp8/choose.py       채널 단위 포맷 선택 방법 A / B / C
run_mixed.py        단계 4 채널 단위 혼합 평가
analyze_pw.py       pointwise / 첫 conv / FC 분포 분석
fp8/pw.py           pointwise FP8 양자화(정확한 덧셈)와 포맷 선택
run_pw.py           pointwise 혼합 정밀도 실험
make_figures.py     발표용 그림 4개 생성 (figures/)
```

## 준비
```
pip install torch torchvision
```

## 단계 0: 비트 정확 모델 검증 (약 1분, CPU)
```
python tests/test_step0.py
```
마지막 줄이 `STEP 0: ALL PASS` 여야 한다.

## 단계 1: FP32 학습과 BN 합치기
```
python train.py        # 200 epoch
python fold_bn.py
```
`fold_bn.py` 의 `float64 max logit diff` 가 1e-6 이하이고 `STEP 1 fold check: PASS` 여야 한다.
이후 실험의 FP32 기준 정확도는 `after fold` 값이다.

## 단계 2: 분포 분석
```
python analyze.py
```

## 단계 3: 동일 포맷 8조합
```
python tests/test_step3.py   # STEP 3 dw_chip check: PASS
python run_uniform.py
```

## 단계 4: 채널 단위 혼합 (방법 A / B / C)
```
python run_mixed.py
```

## 고정 규칙
- mode: 0 = E4M3, 1 = E5M2
- 커널 인덱스 k = 3*행 + 열, 누산 순서 0 -> 8, 패딩 위치 입력은 0
- 양자화: 최근접 반올림(동률은 짝수 코드), 최대 정규값 포화, subnormal 없음
