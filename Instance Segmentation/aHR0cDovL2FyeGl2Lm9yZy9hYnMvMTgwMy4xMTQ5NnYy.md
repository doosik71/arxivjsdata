# Predicting Future Instance Segmentation by Forecasting Convolutional Features

Pauline Luc, Camille Couprie, Yann LeCun, and Jakob Verbeek (2018)

## 🧩 Problem to Solve

본 논문은 비디오의 미래 프레임에 대한 **Future Instance Segmentation(미래 인스턴스 분할)** 문제를 해결하고자 한다. 기존의 비디오 예측 연구들은 주로 미래의 RGB 픽셀 값을 직접 예측하는 방식에 집중해 왔으나, 이는 연산 비용이 높고 시각적으로 흐릿한 결과를 초래하는 경우가 많았다. 또한, 미래의 Semantic Segmentation(의미론적 분할)을 예측하는 연구가 진행되었으나, 이는 동일한 클래스의 객체들을 하나의 덩어리로 묶어 처리할 뿐 개별 객체(Instance)를 구분하지 못한다는 한계가 있다.

자율 주행과 같은 지능형 시스템에서는 개별 객체의 궤적을 추적하고 객체 간의 상호작용 및 변형을 추론하는 능력이 필수적이다. 따라서 본 연구의 목표는 미래 프레임에서 개별 객체를 분리하여 인식하는 Instance Segmentation을 효과적으로 수행할 수 있는 예측 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 미래의 RGB 이미지나 가변적인 인스턴스 라벨을 직접 예측하는 대신, **고정된 크기를 가진 고수준의 Convolutional Feature(합성곱 특징 맵)를 예측**하는 것이다.

구체적으로는 Mask R-CNN의 Backbone인 Feature Pyramid Network(FPN)에서 추출되는 특징 맵을 예측 대상으로 삼는다. 예측된 특징 맵을 Mask R-CNN의 기존 "Detection Head"에 입력하면, 별도의 추가 학습 없이도 미래 프레임의 인스턴스 분할 결과를 얻을 수 있다. 이러한 접근 방식은 다음과 같은 이점을 제공한다.
1. 객체 검출 및 인스턴스 분할에서 발생하는 가변적인 출력 크기 문제를 고정 크기의 특징 공간 내에서 해결함으로써 처리 가능하다.
2. 중간 단계의 CNN 특징 맵은 레이블이 없는 데이터에서도 추출 가능하므로, 학습을 위해 레이블링된 비디오 시퀀스가 필수적이지 않은 자기지도 학습(Self-supervised) 구조를 가진다.
3. 특징 맵을 예측하므로, 동일한 특징을 사용하는 다양한 다운스트림 작업(표면 법선 예측, 바운딩 박스 등)에 범용적으로 적용 가능하다.

## 📎 Related Works

**1. 미래 비디오 예측 (Future Video Prediction)**
기존 연구들은 주로 RGB 값의 직접 예측에 집중했으며, Autoregressive 모델, GAN, Recurrent Network 등이 사용되었다. 최근에는 RGB 대신 추상적인 표현(Semantic level)을 예측하는 것이 더 효과적이라는 결과가 보고되었다. 특히 Luc et al. [1]은 미래의 Semantic Segmentation을 예측하는 S2S 모델을 제안하였으나, 이는 인스턴스 레벨의 구분이 불가능하다는 한계가 있다.

**2. 특징 맵 예측 (Feature Prediction)**
Vondrick et al. [18]은 미래의 CNN 특징을 예측하여 액션 및 객체 출현을 예측하려 했다. 하지만 해당 연구는 이미지 수준의 전역적 레이블(Global labels)을 예측하는 데 그쳤으며, 본 논문은 이를 확장하여 공간적 세부 정보가 유지되는 밀집된(Spatially dense) 특징 맵을 예측함으로써 인스턴스 분할이라는 훨씬 복잡한 작업을 수행한다.

**3. 인스턴스 분할 (Instance Segmentation)**
본 연구는 현재 SOTA 모델인 Mask R-CNN을 기반으로 한다. Mask R-CNN은 객체 검출(Faster R-CNN)에 마스크 예측 분기를 추가하여 정밀한 정렬과 분할을 수행하는 구조이다.

## 🛠️ Methodology

### 전체 파이프라인
본 모델의 전체 흐름은 다음과 같다:
$$\text{과거 프레임들} \rightarrow \text{FPN 특징 추출} \rightarrow \text{F2F 모델 (특징 예측)} \rightarrow \text{예측된 특징 맵} \rightarrow \text{Mask R-CNN Detection Head} \rightarrow \text{미래 인스턴스 마스크}$$

### 주요 구성 요소 및 역할
1. **Mask R-CNN Backbone (FPN):** 입력 이미지 $X$로부터 다양한 해상도의 특징 피라미드 $P_2, \dots, P_5$를 추출한다. $P_l$의 해상도는 $(H/2^l \times W/2^l)$이다.
2. **Feature-to-Feature (F2F) 모델:** 각 FPN 레벨 $l$에 대해 독립적인 예측 네트워크 $\text{F2F}_l$을 구축한다. 이 모델은 시점 $t-\tau$부터 $t$까지의 특징 맵들을 입력받아 시점 $t+1$의 특징 맵을 예측한다.
3. **Detection Head:** 예측된 $\hat{P}_{t+1}$을 입력받아 RoI(Region of Interest)를 제안하고, 각 RoI에 대해 클래스, 바운딩 박스, 이진 마스크를 생성한다.

### F2F 모델 아키텍처
각 $\text{F2F}_l$ 네트워크는 해상도를 유지하는 CNN으로 구현되었으며, 효율적인 시야(Field of View) 확보를 위해 멀티스케일 구조를 채택한다.
- **멀티스케일 구조:** 입력 데이터를 단계적으로 다운샘플링하여 처리한 후, 다시 업샘플링하여 이전 단계의 예측 결과와 결합(Concatenation)하고 정교화(Refinement)하는 과정을 반복한다.
- **Dilated Convolutions:** 계산 효율을 유지하면서 수용 영역(Receptive field)을 넓히기 위해 확장 합성곱(Dilated convolution)을 사용한다.

### 학습 절차 및 손실 함수
- **손실 함수:** 예측된 특징 맵과 실제 정답 특징 맵 사이의 $\ell_2$ 손실(Mean Squared Error)을 최소화한다.
$$\mathcal{L} = \|\hat{P}_{t+1} - P_{t+1}\|_2^2$$
- **학습 순서:** 가장 해상도가 낮은 $P_5$ 모델을 먼저 학습시킨 후, 그 가중치를 $P_4, P_3, P_2$ 모델의 초기값으로 사용하는 전이 학습 방식을 취한다.
- **Autoregressive Fine-tuning:** 다중 타임스텝 예측 시 발생하는 오차 누적 문제를 해결하기 위해, BPTT(Backpropagation Through Time)를 통해 재귀적으로 파인튜닝을 수행한다.

## 📊 Results

### 실험 설정
- **데이터셋:** Cityscapes 데이터셋 (도시 환경의 도로 주행 영상).
- **지표:** 
    - 인스턴스 분할: $AP_{50}$ (IoU 0.5 기준), summary $AP$ (IoU 0.5~0.95 평균).
    - 의미론적 분할: IoU, RI(Probabilistic Rand Index), GCE(Global Consistency Error), VoI(Variation of Information).
- **비교 대상 (Baselines):**
    - **Oracle:** 미래 RGB 프레임에 직접 Mask R-CNN 적용.
    - **Copy:** 마지막 관찰 프레임의 분할 결과를 그대로 사용.
    - **Optical Flow (Shift/Warp):** 광학 흐름을 이용해 마스크를 단순 이동(Shift)하거나 변형(Warp).
    - **Mask H2F:** Mask R-CNN을 직접 수정하여 미래 결과를 예측하도록 파인튜닝.
    - **S2S:** 이전 연구의 Semantic Segmentation 예측 모델.

### 정량적 결과
- **인스턴스 분할 성능:** F2F 방식이 모든 베이스라인을 압도했다. 특히 중기 예측(mid-term, 0.5초 후)에서 $AP_{50}$ 기준 최선의 베이스라인 대비 약 37% 이상의 상대적 성능 향상을 보였다.
- **의미론적 분할 성능:** IoU 관점에서는 S2S 모델과 유사하거나 단기 예측에서 더 우수(61.2%)했으며, 특히 객체의 윤곽선 묘사 능력(RI, VoI, GCE 지표)에서 훨씬 정밀한 결과를 나타냈다.

### 정성적 결과
- **Warp vs F2F:** Warp 방식은 단순한 흐름에 의존하여 객체의 변형을 잡지 못하고 오차가 누적되는 경향이 있으나, F2F는 씬의 역동성을 학습하여 더 정확한 레이아웃을 예측했다.
- **Mask H2F:** 특정 객체 주변에 여러 개의 중복된 마스크를 생성하는 경향이 발견되었는데, 이는 네트워크가 가능한 여러 위치를 동시에 예측하려는 성질 때문으로 분석된다.

## 🧠 Insights & Discussion

### 강점
- **특징 공간의 효율성:** 픽셀 공간이나 라벨 공간이 아닌 고차원 특징 공간에서 예측함으로써, 가변적인 객체 수 문제를 해결하고 의미론적 일관성을 유지할 수 있었다.
- **범용성:** Mask R-CNN의 Detection Head를 그대로 사용하므로, 다른 인식 헤드(예: 표면 법선 예측)만 교체하면 다양한 작업으로 쉽게 확장 가능하다.

### 한계 및 미해결 질문
- **결정론적 예측의 한계:** 모델이 결정론적(Deterministic)으로 설계되어, 보행자의 다리 움직임과 같이 불확실성이 큰 영역에서는 여러 가능성의 평균값을 출력하여 마스크가 뭉개지는(Blurry) 현상이 발생한다. 이는 GAN이나 VAE와 같은 확률적 모델 도입을 통해 해결할 수 있을 것이다.
- **폐색(Occlusion) 처리:** 객체가 다른 객체에 가려지는 상황에 대한 명시적인 모델링이 부족하여, 일부 케이스에서 일관성 없는 마스크가 생성된다.
- **장기 예측의 어려움:** 1.5초 이상의 장기 예측에서는 모든 모델의 성능이 급격히 저하되었으며, 이는 미래 예측 문제의 근본적인 난제임을 보여준다.

## 📌 TL;DR

본 논문은 미래의 인스턴스 분할(Future Instance Segmentation)이라는 새로운 과제를 정의하고, 이를 위해 **Mask R-CNN의 FPN 특징 맵을 예측하는 F2F(Feature-to-Feature) 모델**을 제안하였다. 픽셀이나 라벨을 직접 예측하는 대신 고정 크기의 특징 공간에서 예측을 수행한 후 기존 Detection Head를 적용함으로써, 광학 흐름 기반의 Warp 방식이나 단순 모델 수정 방식보다 훨씬 정밀한 미래 객체 분할 성능을 달성하였다. 이 연구는 고차원 CNN 특징 예측이 복잡한 시각적 추론 작업에 효과적으로 확장될 수 있음을 입증하였다.