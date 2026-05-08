# Medical Image Segmentation Using a U-Net type of Architecture

Eshal Zahra, Bostan Ali, and Wajahat Siddique (2020)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 작업에서 인코더(Encoder)가 추출하는 특징 표현(feature representation)의 질을 향상시키는 문제를 다룬다. 의료 영상 분할은 MRI, CT scan, 초음파 영상 내에서 암 조직이나 결함 부위를 국소화(localization)하는 데 있어 매우 중요한 역할을 한다.

기존의 U-Net과 같은 Encoder-Decoder 구조는 매우 효과적이지만, 본 논문은 인코더의 최하단부인 병목 층(bottleneck layer)에서 명시적인 지도 학습(supervised training) 전략을 결합함으로써, 더 풍부한 시맨틱 정보(semantic information)를 학습하고 이를 통해 분할 성능을 최적화하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 U-Net의 인코더 브랜치 끝단인 병목 층에 **완전 지도 학습 기반의 Fully Connected(FC) 레이어 서브넷(sub-net)**을 도입하는 것이다.

일반적인 U-Net은 전체 네트워크의 최종 출력단에서만 손실 함수를 계산하는 엔드-투-엔드(end-to-end) 방식을 취하지만, 본 제안 방법은 병목 층에서 픽셀 단위의 손실(pixel-wise loss)을 직접 계산하는 FC 레이어를 추가한다. 이를 통해 인코더가 최종 디코더(Decoder)로 정보를 넘겨주기 전에, 병목 표현(bottleneck representation) 자체가 이미 정답지(ground-truth)에 가까운 시맨틱 정보를 포함하도록 강제하며, 결과적으로 디코더가 더 정확한 분할 맵을 생성할 수 있도록 돕는다.

## 📎 Related Works

논문에서는 의료 영상 분할 방법을 크게 두 가지 범주로 구분하여 설명한다.

1. **전통적인 방법(Conventional Methods):** 임계값 기반 방법(thresholding based), 영역 성장 방법(region growing), 클러스터링 기반 방법(clustering based) 등이 있으며, 이는 딥러닝 이전의 표준적인 접근 방식이었다.
2. **딥러닝 기반 방법(Deep Learning Methods):**
    - **U-Net:** 생물학적 현미경 이미지 분할을 위해 제안되었으며, Contracting path(인코더)와 Expanding path(디코더) 및 스킵 연결(skip-connection)을 통해 문맥 정보와 국소화 정보를 동시에 포착한다.
    - **V-Net:** 3D 의료 영상 분할을 위해 제안되었으며, 전경과 배경의 불균형 문제를 해결하기 위해 Dice coefficient 기반의 손실 함수를 사용한다.
    - **PDV-Net (Progressive Dense V-net):** 흉부 CT 영상의 폐엽(pulmonary lobes) 분할을 위해 Dense feature block을 도입하여 사용자 개입 없이 자동 분할을 수행한다.

본 논문은 이러한 기존 구조들의 장점을 수용하면서, 병목 층에 명시적인 감독 신호(supervisory signal)를 제공한다는 점에서 기존의 자가 지도적 성격이 강한 일반적 U-Net 학습 방식과 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조

제안하는 네트워크는 크게 세 가지 구성 요소로 이루어져 있다.

1. **Encoder Part:** 전형적인 CNN 설계로, $3 \times 3$ 컨볼루션 필터와 ReLU 활성화 함수, 그리고 $2 \times 2$ Max Pooling(stride 2) 층으로 구성된다. 이 과정에서 영상의 공간적 크기는 줄어들고 채널 수는 증가하여 고차원의 특징을 추출한다.
2. **Bottleneck Training Part:** 인코더의 최하단에 위치하며, 두 개의 Fully Connected(FC) 레이어로 구성된 서브넷이다. 이 부분은 선형 변환을 통해 입력 이미지로부터 정답 분할 맵을 직접 예측하도록 설계되었다.
3. **Decoder Part:** $2 \times 2$ Up-convolution을 통해 특징 맵의 크기를 다시 키우며, 인코더의 중간 층에서 추출된 특징 맵을 스킵 연결(skip-connection)을 통해 결합(concatenation)한다. 이후 $3 \times 3$ 컨볼루션과 ReLU를 통해 최종 분할 맵을 생성한다.

### 학습 절차 및 손실 함수

본 모델은 두 가지 서로 다른 손실 함수를 사용하여 동시에 학습한다.

- **Bottleneck Sub-net:** 픽셀 단위의 교차 엔트로피 손실(pixel-wise cross entropy loss)을 사용하여 FC 레이어가 정답 맵을 정확히 예측하도록 학습한다.
- **Main U-Net Architecture:** 전체 네트워크의 최종 출력단에서는 $L1$ 손실 함수를 사용하여 학습을 진행한다.

### 평가 지표

모델의 성능을 측정하기 위해 다음과 같은 수식을 사용한다.

$$ \text{Sensitivity} = \frac{N_{tp}}{N_p} $$
$$ \text{Specificity} = \frac{N_{tn}}{N_n} $$
$$ \text{Accuracy} = \frac{N_{tp} + N_{tn}}{N_{tp} + N_{tn} + N_{fn} + N_{fp}} $$

여기서 $N_{tp}$는 True Positive, $N_{tn}$은 True Negative, $N_{fp}$는 False Positive, $N_{fn}$은 False Negative의 개수를 의미한다.

## 📊 Results

제안된 방법은 MRI 및 CT Scan 이미지 데이터셋을 통해 평가되었다. 정량적 결과는 다음과 같다.

| Data | Specificity | Sensitivity | Accuracy |
| :--- | :---: | :---: | :---: |
| MRI | 0.926 | 0.939 | 0.913 |
| CT Scan | 0.961 | 0.972 | 0.976 |

*(참고: 원문 Table 1의 캡션에는 'Oulu-CASIA: Accuracy for six expressions classification'이라고 적혀 있으나, 내부 데이터는 MRI와 CT Scan의 결과이므로 이는 논문 작성상의 오타로 판단되며, 실제 내용은 의료 영상 분할 결과이다.)*

실험 결과, CT Scan 영상에서 특히 높은 정확도($0.976$)와 민감도($0.972$)를 보였으며, MRI 영상에서도 $0.9$ 이상의 높은 성능을 기록하여 제안한 병목 층 지도 학습 방식이 유효함을 보여주었다.

## 🧠 Insights & Discussion

본 논문은 U-Net의 잠재 공간(latent space)인 병목 층에 직접적인 감독 신호를 부여함으로써 인코더의 특징 추출 능력을 강제적으로 향상시킬 수 있음을 시사한다. 이는 딥러닝 모델에서 중간 층의 표현력을 높이기 위해 보조 손실 함수(auxiliary loss)를 사용하는 전략과 맥을 같이 한다.

**비판적 해석 및 한계점:**

1. **비교 대상의 부재:** 저자들은 본 방법이 오리지널 U-Net과 "비교 가능한(comparable)" 혹은 더 나은 결과를 낸다고 주장하지만, 정작 baseline인 오리지널 U-Net과의 정량적 비교 수치(Table)를 제시하지 않았다. 따라서 제안 방법이 구체적으로 얼마나 성능을 향상시켰는지 객관적으로 판단하기 어렵다.
2. **데이터셋 정보 미비:** 사용된 MRI 및 CT Scan 데이터셋의 출처, 이미지의 개수, 클래스의 종류 등 상세 정보가 명시되지 않아 실험의 재현성이 떨어진다.
3. **손실 함수의 선택:** 일반적으로 분할 작업에서는 Dice Loss나 Cross Entropy를 주 손실 함수로 사용하는데, U-Net 전체 학습에 $L1$ loss를 사용한 구체적인 이유나 근거가 설명되지 않았다.

## 📌 TL;DR

본 논문은 U-Net 아키텍처의 병목 층에 Fully Connected 레이어 기반의 지도 학습 서브넷을 추가하여, 인코더가 더 정교한 시맨틱 특징을 학습하도록 유도하는 방법을 제안한다. MRI와 CT 영상 분할 실험에서 높은 정확도를 보였으며, 이는 중간 단계의 명시적 감독이 전체 네트워크의 표현 학습에 긍정적인 영향을 줄 수 있음을 보여준다. 향후 의료 영상의 도메인 특성에 맞는 보조 학습 전략 연구에 참고가 될 수 있다.
