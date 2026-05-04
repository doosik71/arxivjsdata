# Behavior of Keyword Spotting Networks Under Noisy Conditions

Anwesh Mohanty, Adrian Frischknecht, Christoph Gerum, and Oliver Bringmann (2021)

## 🧩 Problem to Solve

본 논문은 인공지능 및 스마트 기기의 핵심 기능인 Keyword Spotting (KWS) 시스템이 고소음 환경(High Noise Conditions)에서 겪는 성능 저하 문제를 해결하고자 한다. 

최근의 KWS 연구들은 저소음 또는 중등도 소음 환경의 데이터셋에서 높은 정확도를 달성하는 다양한 아키텍처를 제안해 왔다. 하지만 실제 환경(예: 심한 교통 소음, 건설 현장 등)에서는 신호 대 잡음비(Signal-to-Noise Ratio, SNR)가 매우 낮아지며, 특히 추론(Inference) 단계에서 발생하는 소음의 특성이 학습(Training) 단계에서 경험하지 못한 종류일 경우 모델의 성능이 급격하게 하락하는 문제가 발생한다.

따라서 본 연구의 목표는 최신 KWS 네트워크들이 다양한 소음 조건에서 어떻게 동작하는지 상세히 분석하고, 특히 학습 시 알 수 없었던 소음이 발생하는 환경에서도 강건하게 동작할 수 있는 효율적인 방법론을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **KWS 네트워크의 소음 특성 분석**: 최신 KWS 모델인 TC-ResNet8과 SCN(SincConv Network)을 대상으로 다양한 소음 수준 및 소음 종류에 따른 성능 변화를 정량적으로 분석하였다.
2. **Modified SCN 아키텍처 제안**: 기존 SCN 모델의 높은 연산 비용(MACs) 문제를 해결하기 위해, 하이퍼파라미터 튜닝과 구조 변경을 통해 정확도는 유지하면서 연산량과 메모리 사용량을 약 절반으로 줄인 최적화 모델을 제안하였다.
3. **Adaptive Batch Normalization 도입**: 테스트 단계에서 학습 시와 다른 분포의 소음이 유입될 때, 학습된 통계값이 아닌 현재 배치의 통계값을 실시간으로 사용하는 Adaptive BatchNorm 기법을 통해 고소음 환경에서의 정확도를 유의미하게 향상시켰다.

## 📎 Related Works

기존의 KWS 연구는 주로 하드웨어 구현을 위한 저전력, 소형 모델 개발에 집중해 왔다.
- **MFCC 기반 접근**: TC-ResNet과 같은 모델은 MFCC(Mel-Frequency Cepstral Coefficients) 특징량을 입력으로 사용하여 매우 높은 정확도와 속도를 달성하였다.
- **Raw Audio 기반 접근**: MFCC 전처리 과정 없이 원시 오디오 데이터를 직접 처리하기 위해 SincNet 기반의 SCN 아키텍처가 제안되었으며, 이는 Sinc-convolution을 통해 효율적으로 특징을 추출한다.

그러나 기존 연구들은 소음 환경에 대한 분석이 부족하거나, 사용한 데이터셋과 지표가 상이하여 최신 모델들의 소음 강건성을 직접적으로 비교하기 어려웠다. 본 논문은 표준 데이터셋을 사용하여 고소음 및 미지의 소음 환경에서의 성능을 체계적으로 분석했다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조 및 비교 모델
본 연구에서는 다음 세 가지 모델을 비교 분석한다.

1. **TC-ResNet8**: MFCC 특징량을 입력으로 받으며, 1차원 Temporal Convolution과 Residual Block을 사용하여 연산량을 줄인 구조이다.
2. **SCN (SincConv Network)**: Raw Audio를 입력으로 받으며, 첫 번째 레이어에서 Sinc-convolution 필터를 사용하여 특정 주파수 대역의 정보를 추출한다. 이후 Depthwise Separable (DS) Convolution 블록과 Global Average Pooling을 거쳐 분류를 수행한다.
3. **Modified SCN**: SCN의 높은 연산 비용을 줄이기 위해 다음과 같은 최적화를 적용하였다.
    - CNN 레이어의 그룹핑(Grouping) 설정을 $(2, 3)$에서 $(4, 8)$로 변경하여 파라미터 수를 감소시켰다.
    - Sinc-convolution 레이어의 Stride를 두 배로 늘려 이후 레이어로 전달되는 피처 맵의 크기를 줄임으로써 전체 MACs를 획기적으로 낮추었다.

### Sinc-convolution의 원리
SCN의 핵심인 Sinc-convolution 필터는 주파수 영역에서 rectangular band-pass filter로 정의된다.
- 주파수 영역 표현:
$$H[f, f_1, f_2] = \text{rect}\left(\frac{f}{f_2}\right) - \text{rect}\left(\frac{f}{f_1}\right)$$
- 시간 영역 표현:
$$h[n, f_1, f_2] = 2f_2 \text{sinc}(2\pi f_2 n) - 2f_1 \text{sinc}(2\pi f_1 n)$$
여기서 $f_1$과 $f_2$는 각각 하한 및 상한 컷오프 주파수로, 필터는 이 두 지점 사이의 정보만을 추출한다. 이 과정 이후에는 $y = \log(|x| + 1)$ 형태의 log-compression 활성화 함수가 적용된다.

### Adaptive Batch Normalization
일반적인 Batch Normalization (BatchNorm)은 학습 단계에서 계산된 이동 평균(Moving Average) 통계값을 테스트 단계에서 고정하여 사용한다. 하지만 테스트 데이터의 소음 분포가 학습 데이터와 크게 다를 경우(Covariate Shift), 고정된 통계값은 오히려 성능 저하를 유발한다.

본 논문에서는 추론 단계에서도 BatchNorm 레이어를 끄지 않고, 현재 입력된 배치의 통계값을 실시간으로 계산하여 정규화에 사용하는 방식을 제안한다. 이를 통해 네트워크가 추론 시점의 소음 특성에 적응(Adaptation)하게 함으로써 성능을 높인다.

## 📊 Results

### 실험 설정
- **데이터셋**: Google Speech Commands Dataset (10개 키워드 + unknown + silence).
- **소음 종류**: White noise, Pink noise, Miscellaneous noise (실제 생활 소음).
- **SNR 범위**: $-5\text{dB}$에서 $+10\text{dB}$까지.
- **평가 시나리오**: 
    1. 학습/검증/테스트 세트에 동일한 종류의 소음이 포함된 경우 (Known Noise).
    2. 학습 시에는 White/Pink noise만 사용하고, 테스트 시에만 Miscellaneous noise를 주입한 경우 (Unknown Noise).

### 주요 결과
1. **소음 강도에 따른 성능**: 모든 모델에서 SNR이 낮아질수록(소음이 심해질수록) 정확도가 하락한다. 특히 SNR이 $-5\text{dB}$에 도달하면 최신 모델들도 약 $10\%$ 이상의 성능 하락을 보인다.
2. **미지 소음의 영향**: 학습 시 경험하지 못한 Miscellaneous noise가 주입되었을 때, SNR $-5\text{dB}$ 환경에서 성능이 파괴적으로(Catastrophically) 떨어진다. 알려진 소음 환경과 비교했을 때 최대 $40\%$의 정확도 차이가 발생하였다.
3. **Adaptive BatchNorm의 효과**: 이 기법을 적용했을 때 고소음 영역에서 성능이 비약적으로 향상되었다. 특히 SNR $-5\text{dB}$에서 TC-ResNet8은 약 $20\%$, SCN 모델들은 약 $10\%$의 정확도 상승을 기록하였다.
4. **Modified SCN 효율성**: Modified SCN은 원본 SCN 대비 MACs를 $18\text{M} \to 7.5\text{M}$으로, 파라미터를 $60\text{k} \to 34.5\text{k}$로 줄였음에도 불구하고 유사한 정확도를 유지하였다.

## 🧠 Insights & Discussion

본 논문은 KWS 모델이 단순한 데이터 증강만으로는 해결할 수 없는 '미지의 고소음 환경'에서의 취약성을 명확히 드러냈다. 

**강점**: 
- 단순한 아키텍처 수정(Stride, Grouping)과 추론 방식의 변경(Adaptive BN)만으로 연산 효율성과 강건성을 동시에 확보하였다.
- 특히 Adaptive BN은 추가적인 학습 시간이나 복잡한 모델 구조 변경 없이도 성능을 높일 수 있는 매우 실용적인 해결책임을 입증하였다.

**한계 및 논의**:
- 저자들은 모델의 크기를 키우거나 더 방대한 소음 데이터셋으로 학습시키는 방법이 성능을 높일 수는 있겠지만, 이는 '소형 풋프린트(Small Footprint)'라는 KWS의 본질적인 목표에 어긋난다고 지적한다.
- Adaptive BN은 배치 단위의 통계량을 사용하므로, 실제 실시간 추론 시 배치 크기가 1인 경우(Single sample inference) 어떻게 적용할 것인지에 대한 구체적인 구현 방법이 명시되지 않았다.

## 📌 TL;DR

본 연구는 고소음 환경에서 KWS 네트워크의 성능 저하 문제를 분석하고, 이를 해결하기 위해 **연산량을 절반으로 줄인 Modified SCN 아키텍처**와 **미지 소음 적응을 위한 Adaptive BatchNorm 기법**을 제안하였다. 특히 Adaptive BN은 $-5\text{dB}$의 극한 소음 환경에서 TC-ResNet8의 정확도를 약 $20\%$ 향상시키는 성과를 거두었으며, 이는 향후 저전력 임베디드 환경의 강건한 음성 인식 시스템 구축에 중요한 기초 자료가 될 것으로 보인다.