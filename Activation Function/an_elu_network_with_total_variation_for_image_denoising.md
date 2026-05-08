# An ELU Network with Total Variation for Image Denoising

Tianyang Wang, Zhengrui Qin, and Michelle Zhu (2017)

## 🧩 Problem to Solve

본 논문은 가우시안 노이즈가 추가된 이미지로부터 원래의 깨끗한 이미지를 복원하는 이미지 디노이징(Image Denoising) 문제를 해결하고자 한다. 일반적으로 노이즈가 섞인 이미지는 $y = x + v$로 모델링되며, 여기서 $x$는 잠재된 깨끗한 이미지, $v$는 가산 가우시안 백색 잡음(Additive Gaussian White Noise)이다.

최근 딥러닝 기반의 잔차 학습(Residual Learning) 방식이 우수한 성능을 보이고 있으나, 본 연구는 기존의 합성곱 신경망(CNN)에서 널리 사용되는 활성화 함수인 ReLU(Rectified Linear Unit)와 손실 함수를 재검토함으로써 디노이징 성능을 더욱 향상시키는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같이 세 가지로 요약할 수 있다.

1. **ELU 활성화 함수의 적합성 분석**: ELU(Exponential Linear Unit)가 ReLU보다 이미지 디노이징 작업, 특히 잔차 매핑(Residual Mapping)을 학습하는 데 더 적합함을 ASM(Angular Second Moment) 에너지 관점에서 분석하고 증명하였다.
2. **새로운 계층 조합 설계**: Batch Normalization(BN)과 ELU를 직접 결합할 경우 성능이 저하되는 문제를 해결하기 위해, 그 사이에 $1 \times 1$ Convolution 레이어를 삽입한 'Conv-ELU-Conv-BN' 구조의 기본 블록을 제안하였다.
3. **TV 정규화 기반의 손실 함수**: 기존 $L_2$ 손실 함수에 Total Variation(TV) 정규화 항을 추가하여, 학습 과정에서 이미지의 구조적 특성을 보존하고 디노이징 효과를 극대화하였다.

## 📎 Related Works

이미지 디노이징 방법론은 크게 두 가지 범주로 나뉜다.

- **이미지 사전 모델링 기반 방법(Image Prior Modeling based)**: BM3D, LSSC, EPLL, WNNM 등이 이에 해당하며, 이미지의 통계적 특성이나 사전 지식을 활용하여 복원한다.
- **판별 학습 기반 방법(Discriminative Learning based)**: MLP, CSF, DGCRF, NLNet, TNRD 등이 있으며, 최근에는 DnCNN과 같은 깊은 잔차 학습(Deep Residual Learning) 방식이 매우 뛰어난 성과를 거두고 있다.

본 논문은 특히 ReLU를 사용하는 DnCNN의 구조적 한계를 지적하며, 활성화 함수를 ELU로 교체하고 TV 정규화를 도입함으로써 기존의 판별 학습 기반 접근 방식을 개선하고자 한다.

## 🛠️ Methodology

### 1. Exponential Linear Unit (ELU)

본 논문에서는 활성화 함수로 ELU를 채택한다. ELU는 다음과 같이 정의된다.

$$f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \le 0 \end{cases}$$

여기서 $\alpha$는 음수 입력에 대한 포화 수준을 조절하는 파라미터이다. ELU는 ReLU와 달리 음수 값을 완전히 0으로 만들지 않으므로, 유닛 활성화의 평균값이 0에 가깝게 유지된다. 이는 학습 속도를 높이고 노이즈에 대한 시스템의 강건성(Robustness)을 향상시킨다.

### 2. ELU 사용의 동기: ASM 분석

연구진은 ASM(Angular Second Moment)을 통해 ELU의 효율성을 분석하였다. ASM은 다음과 같이 계산된다.

$$\text{ASM} = \sum_{i,j=0}^{N-1} P_{i,j}^2$$

여기서 $P_{i,j}$는 노이즈 매핑의 회색조 동시 발생 행렬(GLCM)의 원소이다. 일반적으로 노이즈가 많은 이미지일수록 ASM 값이 낮다. 잔차 학습의 목표는 최대한 많은 노이즈를 포함하는 노이즈 매핑 $v$를 학습하는 것이므로, ASM 값이 낮은 활성화 함수를 선택하는 것이 유리하다. 실험 결과, ELU를 사용했을 때 ReLU보다 더 낮은 ASM 값을 가질 확률이 높았으며, 이는 ELU가 더 효과적으로 노이즈를 제거할 수 있음을 시사한다.

### 3. TV Regularized $L_2$ Loss

이미지 디노이징에서 TV 최소화는 널리 사용되는 기법이다. 본 논문은 이를 학습 단계에 도입하여 다음과 같은 손실 함수를 정의하였다.

$$L = \frac{1}{2N} \sum_{i=1}^{N} ||R - (y_i - x_i)||^2 + \beta \text{TV}(y_i - R)$$

여기서 $R$은 네트워크가 학습한 노이즈 매핑이며, $\text{TV}$ 항은 다음과 같이 계산된다.

$$\text{TV}(u) \approx \sum_{i,j} \sqrt{(\nabla_x u)_{i,j}^2 + (\nabla_y u)_{i,j}^2}$$

첫 번째 항($L_2$ loss)이 노이즈 매핑을 학습한다면, 두 번째 항($\text{TV}$)은 결과 이미지의 평활도를 조절하여 추가적인 디노이징 효과를 부여한다. $\beta$ 값은 학습 에포크에 따라 가변적으로 업데이트하여 최적의 결과를 얻었다.

### 4. 네트워크 아키텍처

전체 구조는 VGG-19 네트워크를 기반으로 하며, Fully Connected 레이어와 Pooling 레이어는 제외되었다.

- **기본 블록**: 'Conv $\rightarrow$ ELU $\rightarrow$ Conv $\rightarrow$ BN' 순서로 구성된 블록이 15회 반복된다.
- **$1 \times 1$ Convolution의 역할**: ELU와 BN을 직접 연결하면 성능이 저하된다는 선행 연구에 따라, 그 사이에 $1 \times 1$ Convolution 레이어를 배치하였다. 이는 BN과 ELU의 직접적인 결합을 방지함과 동시에 결정 함수의 비선형성을 증가시키는 역할을 한다.
- **필터 크기**: 첫 번째 Conv는 $3 \times 3$ 필터를 사용하고, 두 번째 Conv는 $1 \times 1$ 필터를 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: BSD500(학습), BSD68 및 Set12(테스트).
- **노이즈 레벨**: $\sigma = 15, 25, 50$의 특정 레벨 및 $[0, 55]$ 범위의 랜덤 레벨(Blind denoising).
- **지표**: PSNR(Peak Signal-to-Noise Ratio)을 사용하여 성능을 평가하였다.
- **비교 대상**: BM3D, MLP, EPLL, LSSC, CSF, WNNM, DGCRF, TNRD, NLNet, DnCNN 등.

### 주요 결과

- **정량적 결과**:
  - 그레이스케일 이미지 및 컬러 이미지 모두에서 DnCNN보다 약 $0.1\text{dB}$ 정도 높은 PSNR을 기록하며 가장 우수한 성능을 보였다.
  - 특히 $\sigma = 50$인 고노이즈 상황에서 BM3D 대비 $0.7\text{dB}$ 향상된 결과를 보였다.
- **정성적 결과**:
  - 시각적 비교 결과, DnCNN에 비해 이미지의 세부 디테일(Detail)을 더 잘 보존하는 것으로 나타났다.
  - 또한 TV 정규화를 사용했음에도 불구하고 배경이 과도하게 뭉개지는 오버-스무딩(Over-smoothing) 현상이 완화되었다.

## 🧠 Insights & Discussion

본 논문은 단순히 새로운 아키텍처를 제안한 것이 아니라, **에너지 관점(ASM)**에서 활성화 함수의 선택이 디노이징 성능에 미치는 영향을 분석했다는 점에서 학술적 가치가 있다. 특히 ELU가 잔차 매핑의 에너지 수준을 낮춰 더 많은 노이즈를 캡처할 수 있게 한다는 분석은 매우 설득력이 있다.

또한, 딥러닝 모델에서 흔히 발생하는 BN과 특정 활성화 함수 간의 상충 관계를 $1 \times 1$ Convolution이라는 단순하면서도 효과적인 구조로 해결한 점이 인상적이다. TV 정규화의 도입은 전통적인 영상 처리 기법과 현대적인 딥러닝 기법을 성공적으로 결합하여, 단순한 $L_2$ 손실 함수만으로는 달성하기 어려운 구조적 보존 능력을 확보하였다.

다만, ELU의 지수 연산으로 인해 ReLU 대비 계산 복잡도가 증가한다는 점이 언급되었으나, 이에 대한 실제 추론 시간(Inference Time)의 정량적 비교 데이터는 제시되지 않아 실제 적용 시의 오버헤드를 정확히 판단하기 어렵다.

## 📌 TL;DR

본 논문은 이미지 디노이징을 위해 **ELU 활성화 함수**와 **TV 정규화**를 결합한 CNN 모델을 제안하였다. ELU가 노이즈 잔차를 더 잘 학습한다는 점을 ASM 분석으로 증명하였고, 'Conv-ELU-Conv-BN' 구조를 통해 BN과의 결합 문제를 해결하였다. 실험 결과, 기존 SOTA 모델인 DnCNN보다 높은 PSNR과 더 나은 디테일 보존 능력을 보여주었으며, 이는 향후 활성화 함수 및 손실 함수 최적화 연구에 중요한 기초 자료가 될 것으로 보인다.
