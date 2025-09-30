# Depthwise Separable Convolutional ResNet with Squeeze-and-Excitation Blocks for Small-footprint Keyword Spotting

Menglong Xu, Xiao-Lei Zhang

## 🧩 Problem to Solve

키워드 스포팅(Keyword Spotting, KWS)은 오디오 스트림에서 미리 정의된 키워드를 탐지하는 기술로, 특히 메모리 공간이 작고 계산 능력이 낮은 저사양 디바이스에서 고정밀 성능을 유지하는 것이 어려운 문제입니다. 기존 컨볼루션 신경망(CNN) 기반 KWS 모델들은 우수한 성능을 달성하기 위해 수십만 개의 파라미터를 필요로 하여 소형 디바이스에 적용하기에 비효율적입니다. 이 연구의 목표는 이러한 제약 조건을 충족하면서도 높은 정확도를 유지하는, 더 작은 메모리 공간을 갖는 효율적인 KWS 모델을 개발하는 것입니다.

## ✨ Key Contributions

- **DS-ResNet (Depthwise Separable Convolution based ResNet) 제안:** 소규모 키워드 스포팅 문제 해결을 위해 깊이별 분리 컨볼루션(depthwise separable convolution)과 잔차 연결(residual connections)을 기반으로 하는 효율적인 모델을 제안했습니다.
- **파라미터 및 계산량 대폭 감소:** 표준 컨볼루션을 깊이별 분리 컨볼루션으로 대체하여 성능 저하 없이 모델의 파라미터 수와 계산 비용을 크게 줄였습니다.
- **Squeeze-and-Excitation (SE) 블록 적용:** 첫 번째 컨볼루션 레이어의 출력 피처 맵에 Squeeze-and-Excitation 블록을 적용하여 채널별 피처 응답을 적응적으로 재조정함으로써, 파라미터 수를 크게 늘리지 않고도 모델 성능을 추가적으로 향상시켰습니다.
- **최첨단 성능 달성:** Google Speech Commands 데이터셋에서 기존 최첨단 모델들과 비교하여 훨씬 적은 파라미터 수로 동등하거나 더 우수한 분류 에러율을 달성했습니다 (예: 72K 파라미터로 3.29% 에러율, 10K 파라미터로 3.97% 에러율).

## 📎 Related Works

- **Large Vocabulary Continuous Speech Recognition (LVCSR) [1, 2] 및 Keyword/Filler HMMs [3]:** 높은 메모리 및 계산 비용으로 저사양 디바이스에 부적합합니다.
- **DNN-based KWS (DeepKWS [4]):** HMM 기반 방법보다 개선되었지만, 음성의 지역적 시간적/스펙트럼 상관관계를 고려하지 않습니다.
- **CNN for KWS [5]:** DNN보다 작은 메모리 공간으로 더 나은 성능을 보였지만, 수용장(receptive field) 크기가 제한적입니다.
- **ResNet-based KWS [6]:** 확장 컨볼루션(dilated convolution)을 사용하여 수용장을 확장했지만, 여전히 수십만 개의 파라미터를 필요로 합니다.
- **TDNN, Attention, TCN 기반 KWS [7, 8, 9]:** 파라미터를 줄이기 위한 다양한 아키텍처가 제안되었으나 여전히 많은 파라미터가 필요합니다.
- **MobileNet을 KWS에 적용 [10]:** 깊이별 분리 컨볼루션을 사용하지만, ReLU 활성화 함수를 많이 사용할 경우 모델의 표현 능력이 저해될 수 있으며(MobileNetV2 [13]에서 지적), 그래디언트 전파 효율성이 떨어지는 문제가 있습니다.
- **Squeeze-and-Excitation Networks [14]:** 채널 간 상호 의존성을 모델링하여 채널별 피처 응답을 재조정하는 블록입니다.

## 🛠️ Methodology

1. **전체 네트워크 구조 (DS-ResNet):**

   - **초기 컨볼루션 레이어 (Conv):** 표준 편향이 없는 컨볼루션 레이어로 시작합니다.
   - **Squeeze-and-Excitation (SE) 블록:** 초기 Conv 레이어의 출력에 적용되어 채널별 피처 맵의 가중치를 재조정합니다.
   - **잔차 블록 체인:** SE 블록의 출력은 여러 개의 잔차 블록 체인의 입력이 됩니다. 각 잔차 블록은 두 개의 깊이별 분리 컨볼루션(DS-Conv) 레이어로 구성됩니다.
   - **추가 DS-Conv 레이어:** 잔차 블록 체인 다음에 독립적인 비잔차 깊이별 분리 컨볼루션 레이어가 이어집니다.
   - **평균 풀링 (Avg-pool) 및 Softmax:** 모델의 출력은 평균 풀링 레이어와 완전 연결된 Softmax 레이어를 거쳐 최종 분류 결과를 생성합니다.
   - $(d_w, d_h)$ 컨볼루션 확장(dilation)을 사용하여 깊이별 분리 컨볼루션 레이어의 수용장을 증가시켰습니다.

2. **깊이별 분리 컨볼루션 (Depthwise Separable Convolution, DS-Conv):**

   - 표준 컨볼루션을 두 단계로 분리하여 파라미터와 계산량을 줄입니다.
   - **깊이별 컨볼루션 (depth-Conv):** 각 입력 채널에 대해 단일 필터를 적용하여 공간적 특징을 학습합니다.
     - 계산 비용: $C^{(\text{depth-Conv})} = 1 \times D_K \times D_K \times H_{\text{in}} \times W_{\text{in}} \times C_{\text{in}}$
     - 파라미터 수: $S^{(\text{depth-Conv})} = 1 \times D_K \times D_K \times C_{\text{in}}$
   - **점별 컨볼루션 (point-Conv):** $1 \times 1$ 컨볼루션을 사용하여 깊이별 컨볼루션의 출력을 채널 차원에서 결합합니다.
     - 계산 비용: $C^{(\text{point-Conv})} = C_{\text{in}} \times 1 \times 1 \times H_{\text{in}} \times W_{\text{in}} \times C_{\text{out}}$
     - 파라미터 수: $S^{(\text{point-Conv})} = C_{\text{in}} \times 1 \times 1 \times C_{\text{out}}$
   - 예시: 출력 채널 $C_{\text{out}} = 64$, 커널 크기 $D_K = 3$일 때 표준 컨볼루션 대비 계산 비용과 모델 크기가 약 $1/8$로 줄어듭니다.

3. **Squeeze-and-Excitation (SE) 블록:**

   - 채널 간 의존성을 모델링하여 채널별 피처 응답을 동적으로 재조정합니다.
   - **Squeeze 연산:** 전역 평균 풀링(average pooling)을 통해 각 채널의 공간적 차원을 압축하여 채널 디스크립터를 생성합니다.
   - **Excitation 연산:** 두 개의 완전 연결(fully-connected) 레이어(ReLU와 Sigmoid 활성화 함수 사용)를 통해 채널별 가중치를 생성합니다. 이 두 레이어 사이의 차원은 하이퍼파라미터 $\alpha=2^{-4}$에 의해 조절됩니다.
   - 최종 출력은 SE 블록의 입력 피처 맵에 Excitation 연산에서 생성된 채널 가중치를 곱하여 얻어집니다.

4. **모델 구현 및 구성:**
   - **DS-ResNet18:** 가장 높은 정확도를 목표로 하며, 7개의 잔차 블록(총 15개의 DS-Conv 레이어)과 64개의 채널을 사용합니다. 약 72K 파라미터, 285M 곱셈 연산.
   - **DS-ResNet14:** 모델 크기 감소를 위해 5개의 잔차 블록(총 11개의 DS-Conv 레이어)과 32개의 채널을 사용합니다. SE 레이어 뒤에 $2 \times 2$ 평균 풀링 레이어를 추가하여 수용장을 확장했습니다. 약 15.2K 파라미터, 15.7M 곱셈 연산.
   - **DS-ResNet10:** 가장 작은 모델로, 7개의 분리형 컨볼루션 레이어(잔차 연결 없음)와 32개의 채널을 사용합니다. SE 레이어 뒤에 $4 \times 2$ 평균 풀링 레이어를 추가했습니다. 약 10K 파라미터, 5.8M 곱셈 연산.

## 📊 Results

- **데이터셋:** Google Speech Commands Dataset 버전 1을 사용 (10개의 키워드, 20개의 필러, 배경 노이즈). 40차원 Mel-frequency cepstrum coefficient (MFCC) 특징을 추출했습니다.
- **첫 번째 실험 (ResNet [6] 설정 준수):**
  - **DS-ResNet18:** 72K 파라미터로 3.29%의 에러율을 달성. ResNet15 (238K 파라미터, 4.20% 에러율) 대비 파라미터 수가 $1/3$에 불과하면서도 21.7%의 상대적 에러율 감소를 보였습니다.
  - **DS-ResNet14:** 15.2K 파라미터로 4.12%의 에러율을 달성. 유사한 모델 크기의 ResNet8-narrow (19.9K 파라미터, 9.90% 에러율) 대비 58.4%의 상대적 에러율 감소를 보였습니다.
  - **DS-ResNet10:** 10K 파라미터로 4.76%의 에러율을 달성했습니다.
  - **SE 블록 효과:** SE 블록을 사용한 DS-ResNet18 (3.29%)은 SE 블록을 사용하지 않은 DS-ResNet18-n (3.45%)보다 우수하여 SE 블록의 효과를 입증했습니다. 그러나 각 깊이별/점별 컨볼루션 레이어 뒤에 SE 블록을 추가하는 것은 성능 향상으로 이어지지 않았습니다 (DS-ResNet18-d: 3.54%, DS-ResNet18-p: 3.67%).
- **두 번째 실험 (DenseNet-BiLSTM [15], tdnn-swsa [16] 설정 준수):**
  - **DS-ResNet18:** 72K 파라미터로 2.32%의 에러율을 달성. DenseNet-BiLSTM (250K 파라미터, 2.5% 에러율)과 유사한 성능을 파라미터 수의 $1/3$만으로 달성했습니다.
  - **DS-ResNet14:** 15.2K 파라미터로 2.84%의 에러율을 달성. tdnn-swsa (112K 파라미터, 4.19% 에러율) 대비 32.2%의 상대적 에러율 감소를 보였습니다.
  - **DS-ResNet10:** 10K 파라미터로 3.97%의 에러율을 달성. 저자원 환경에서 tdnn-swsa와 경쟁력 있는 성능을 보였습니다.

## 🧠 Insights & Discussion

이 연구는 깊이별 분리 컨볼루션과 Squeeze-and-Excitation 블록을 ResNet 아키텍처와 결합함으로써 소규모 키워드 스포팅 문제에 대한 매우 효과적인 솔루션을 제시했습니다. 주요 통찰은 다음과 같습니다:

- **효율적인 모델 설계:** 깊이별 분리 컨볼루션은 모델의 파라미터 수와 계산 복잡성을 크게 줄이는 데 결정적인 역할을 하여, 성능 저하 없이 소형 풋프린트 모델을 가능하게 합니다. 이는 저전력 및 저메모리 디바이스에 매우 중요합니다.
- **SE 블록의 전략적 활용:** Squeeze-and-Excitation 블록은 모델의 표현 능력을 향상시키지만, 그 배치에 따라 효과가 달라짐을 보여주었습니다. 첫 번째 컨볼루션 레이어 출력에 배치하는 것이 가장 효과적이었으며, 모든 깊이별/점별 컨볼루션 뒤에 추가하는 것은 오히려 성능을 저하시킬 수 있음을 발견했습니다. 이는 SE 블록의 이점을 극대화하기 위한 신중한 아키텍처 설계의 중요성을 시사합니다.
- **우수한 성능-효율성 균형:** 제안된 DS-ResNet 모델들은 다양한 크기(10K에서 72K 파라미터)에서 기존 최첨단 모델들 대비 파라미터 대비 월등히 우수한 성능을 달성하여, 다양한 리소스 제약 조건에 맞는 유연한 솔루션을 제공합니다.
- **한계:** 본 논문에서 명시적인 한계는 다루지 않았지만, SE 블록의 배치가 성능에 미치는 영향은 향후 더 깊은 연구가 필요할 수 있습니다. 또한, 매우 작은 모델(DS-ResNet10)에서 잔차 연결을 제거한 것은 특정 모델 크기 이하에서는 잔차 연결의 이점이 줄어들 수 있음을 암시합니다.

## 📌 TL;DR

**문제:** 저사양 디바이스를 위한 적은 메모리와 계산량으로 고정밀 키워드 스포팅 모델을 개발해야 합니다.
**방법:** 표준 컨볼루션을 깊이별 분리 컨볼루션(Depthwise Separable Convolution)으로 대체하고, 초기 컨볼루션 레이어 출력에 Squeeze-and-Excitation (SE) 블록을 적용한 ResNet 기반 모델(DS-ResNet)을 제안했습니다.
**발견:** DS-ResNet은 Google Speech Commands 데이터셋에서 파라미터 수를 획기적으로 줄이면서도 최신 기술 대비 우수하거나 동등한 성능을 달성했습니다 (예: 72K 파라미터로 3.29% 에러율, 10K 파라미터로 3.97% 에러율). SE 블록은 전략적으로 배치될 경우 성능 향상에 크게 기여합니다.
