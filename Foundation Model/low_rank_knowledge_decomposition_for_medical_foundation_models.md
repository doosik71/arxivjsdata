# Low-Rank Knowledge Decomposition for Medical Foundation Models

Yuhang Zhou, Haolin Li, Siyuan Du, Jiangchao Yao, Ya Zhang, Yanfeng Wang (2024)

## 🧩 Problem to Solve

최근 대규모 데이터셋으로 사전 학습된 의료 파운데이션 모델(Medical Foundation Models)이 등장하며 일반적인 특징 추출 능력이 크게 향상되었다. 그러나 이러한 범용 모델들은 특정 의료 작업(task-specific tasks)에서의 성능이 해당 작업에 특화된 개별 모델보다 열세라는 문제가 있다. 이는 범용성과 특수성(specialization)을 동시에 확보하는 것이 어렵기 때문이며, 데이터의 규모와 다양성이 증가할수록 서로 다른 도메인 지식 간의 충돌로 인해 특수성이 오히려 저하되는 경향이 나타난다.

또한, 파운데이션 모델의 규모가 계속해서 커짐에 따라 실제 의료 현장에 배포할 때 발생하는 계산 비용과 리소스 소모가 지나치게 높다는 점이 실무적인 진입 장벽이 된다. 따라서 본 논문의 목표는 거대한 파운데이션 모델을 여러 개의 경량화된 전문가 모델(lightweight expert models)로 분해하여, 각 전문가 모델이 특정 진료과나 도메인에 특화된 성능을 가지면서도 배포 비용을 획기적으로 낮추는 '지식 분해(Knowledge Decomposition)' 방법론을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 파운데이션 모델의 지식을 공유 지식(common knowledge)과 작업 특화 지식(task-specific knowledge)으로 명시적으로 분리하여 분해하는 것이다. 이를 위해 저자들은 **LoRKD (Low-Rank Knowledge Decomposition)** 프레임워크를 제안하였다.

LoRKD의 중심 설계는 크게 두 가지이다. 첫째, LoRA(Low-Rank Adaptation)에서 영감을 얻은 **Low-Rank Expert Modules**를 통해 파라미터 효율적인 특화 지식 저장소를 구축하는 것이다. 둘째, **Efficient Knowledge Separation (EKS) Convolution**을 도입하여, 단 한 번의 순전파(forward propagation) 과정에서 여러 작업의 그래디언트를 명시적으로 분리하여 각 전문가 모듈에 전달함으로써 학습 효율성을 극대화하는 것이다. 결과적으로 이 방법은 공유 백본은 공통 지식을 학습하고, 전문가 모듈은 각 작업의 특화 지식을 학습하게 하여 모델의 전문성과 전이 가능성(transferability)을 동시에 높인다.

## 📎 Related Works

**Knowledge Distillation (KD)**은 교사 모델의 지식을 학생 모델로 전달하는 효율적인 방법이지만, 본 연구와 같이 지식을 여러 전문가 모델로 '분해'하는 것과는 목적이 다르다. **Multi-Task Learning (MTL)**은 여러 관련 작업을 하나의 모델로 해결하여 일반적인 특징 추출기를 학습시키는 데 집중한다. 그러나 파운데이션 모델 수준의 다양한 작업들은 서로 간의 차이가 매우 크기 때문에, 단순히 공통 지식만을 추구하는 MTL 방식으로는 특수성을 확보하기 어렵다.

최근 자연어/자연 이미지 분야에서 제안된 **KF (Knowledge Factorization)**는 상호 정보량(mutual information)을 조절하여 공통 지식 네트워크(CKN)와 작업 특화 네트워크(TSN)로 모델을 분해하는 시도를 하였다. 하지만 본 논문의 실험 결과, KF 방식은 의료 시나리오에서 효과가 미미했으며, 배포 시 CKN과 TSN을 동시에 운영해야 하므로 효율성이 떨어진다는 한계가 있다. LoRKD는 이러한 한계를 극복하기 위해 저차원 행렬 구조와 파라미터 융합(parameter fusion) 메커니즘을 사용하여 더 효율적인 분해를 수행한다.

## 🛠️ Methodology

### 전체 시스템 구조

LoRKD는 학습 단계에서 하나의 공유 백본 $F_s$와 $T$개의 전문가 모듈 $\{E_1, ..., E_T\}$로 구성된다. 학습이 완료된 후에는 공유 백본의 파라미터와 특정 전문가 모듈의 파라미터를 융합하여, 원래의 파운데이션 모델과 동일한 크기를 가지면서도 특정 작업에 특화된 경량 전문가 모델을 생성하여 배포한다.

### Low-Rank Expert Modules

각 컨볼루션 층의 공유 가중치를 $W_0 \in \mathbb{R}^{C_{out} \times C_{in} \times k \times k}$라고 할 때, $t$번째 작업에 대한 저차원 팩터 $B_t \in \mathbb{R}^{C_{out} \times r \times k \times k}$와 $A_t \in \mathbb{R}^{r \times C_{in} \times k \times k}$를 도입한다. 여기서 $r$은 rank를 의미한다.
특정 작업 $t$에 대한 컨볼루션 연산 $g_t$는 다음과 같이 정의된다:
$$g_t = (W_0 + B_t A_t)h_t$$
여기서 $h_t$는 입력 특징 맵이다. 기존 LoRA와 달리, 공통 지식을 담고 있는 $W_0$ 역시 저차원 팩터들과 함께 업데이트된다.

### Efficient Knowledge Separation (EKS) Convolution

미니배치 내에 $T$개의 서로 다른 작업이 섞여 있을 때, 각 작업마다 별도의 순전파를 수행하는 것은 매우 비효율적이다. 이를 해결하기 위해 EKS Convolution은 파라미터 집계(parameter aggregation) 방식을 사용한다.
입력 특징 맵 $h$와 작업 라벨 $M \in \mathbb{R}^{B \times T}$ (one-hot vector)가 주어졌을 때, 현재 반복 회차에서 사용할 가중치 $W'$를 다음과 같이 계산한다:
$$g = (W_0 + \sum_{i=1}^{T} (g_{BA} \odot M)_i)h = W'h$$
여기서 $g_{BA}$는 모든 전문가 모듈의 가중치를 포함하는 텐서이며, $\odot$는 아다마르 곱(Hadamard product)이다. 이 연산을 위해 저자들은 Group Convolution (GConv) 개념을 도입하여, $W'$를 $B$개의 그룹으로 리셰이프(reshape)함으로써 기존 딥러닝 라이브러리에서 효율적으로 구현하였다.

### 학습 목표 및 손실 함수

파운데이션 모델 $F_p$로부터 지식을 전이하기 위해 Task Knowledge Transfer Loss ($\mathcal{L}_{transfer}$)를 도입하며, 이는 파운데이션 모델의 특징 $f^b_i$와 분해 모델의 특징 $f^d_i$ 사이의 KL 발산(Kullback-Leibler divergence)으로 계산된다. 또한, 작업별 분류 헤드를 통해 작업 수준의 감독(supervision)을 위한 교차 엔트로피 손실 $\mathcal{L}_{CE}$를 추가한다. 전체 손실 함수는 다음과 같다:
$$\mathcal{L}_{total} = \frac{1}{B} \sum_{t=1}^{T} \sum_{i=1}^{B_t} \left( \mathcal{L}_{CE}(y^t_i, p^d_i) + \beta \alpha^2 \mathcal{L}_{KL}(f^b_i, f^d_i) \right)$$
여기서 $\beta$는 하이퍼파라미터이며 $\alpha$는 온도(temperature) 파라미터이다.

### Task Knowledge Switch

배포 시에는 $W_t = W_0 + B_t A_t$와 같이 파라미터를 융합하여 사용한다. 만약 다른 작업 $t'$로 지식을 전환해야 한다면, $W_0 = W_t - B_t A_t$를 통해 공유 가중치를 복구한 후 다시 $W_{t'} = W_0 + B_{t'} A_{t'}$를 계산하여 간단히 전환할 수 있다.

## 📊 Results

### 실험 설정

- **데이터셋**: Radimagenet (1.35M 이미지, 11개 작업), MedMnist (705K 이미지, 10개 작업), Med-MT (119K 이미지, 8개 작업)를 사용하여 사전 학습 및 분해를 수행하였다.
- **전이 가능성 평가**: 분해된 전문가 모델들을 COVID, BTC, AD, Mura, AUITD, HAM10000, DET10 등 7개의 다운스트림 데이터셋에서 평가하였다.
- **비교 대상**: Baseline, STL, MTL, STL-KD, MTL-KD, MoCo-MTL, Aligned-MTL, 그리고 KF를 비교군으로 설정하였다.

### 주요 결과

- **분해 성능**: 세 가지 사전 학습 데이터셋 모두에서 LoRKD는 KF보다 우수한 평균 정확도를 보였으며, 특히 파라미터 수는 KF의 절반 이하로 유지하면서도 가장 높은 성능을 기록하였다.
- **전이 가능성**: 다운스트림 데이터셋 실험 결과, LoRKD로 분해된 전문가 모델들이 Baseline 및 기존 MTL/STL 기반 방법들보다 훨씬 높은 성능을 보였다. 이는 공통 지식과 특화 지식을 모두 보유했기 때문으로 분석된다.
- **비용 효율성**: 파운데이션 모델 대비 파라미터 수와 FLOPs를 획기적으로 줄였으며, 배포 시에는 파라미터 융합을 통해 Baseline과 동일한 수준의 비용으로 운영 가능하다.
- **특수성 분석**: Grad-CAM 시각화 결과, 파운데이션 모델은 전반적으로 넓은 영역을 주목하는 반면, LoRKD 전문가 모델은 실제 병변 부위(Ground-Truth)에 더 정밀하게 집중하는 '강한 특수성'을 보였다.
- **지식 얽힘 해제**: Mutual Information Gap (MIG) 점수를 측정한 결과, LoRKD가 다른 방법론보다 높은 점수를 기록하여 지식 분해(disentanglement)가 성공적으로 이루어졌음을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 파운데이션 모델이 가진 '범용성'이 때로는 특정 작업에서의 '성능 저하'를 야기한다는 점을 정확히 짚어냈으며, 이를 해결하기 위해 모델을 물리적으로 분해하는 접근 방식을 제안하였다. 특히 LoRA의 저차원 구조를 활용하여 파라미터 증가를 억제하면서도, EKS Convolution이라는 효율적인 구현 방식을 통해 학습 속도 문제를 해결한 점이 돋보인다.

**강점**으로는 의료 데이터의 특성상 도메인 간 간섭이 심하다는 점을 명시적 그래디언트 분리(explicit gradient separation)를 통해 해결하여, 단순한 MTL보다 훨씬 높은 특수성을 확보했다는 점이다. 또한, 배포 시 파라미터 융합을 통해 추가적인 추론 오버헤드가 없도록 설계한 점이 실용적이다.

**한계 및 논의사항**으로는, 제안된 방법이 ResNet50 및 ShuffleNetV2와 같은 CNN 기반 구조에서 검증되었으나, 최근 의료 영상 분야에서도 많이 사용되는 Vision Transformer (ViT) 기반의 거대 모델에서도 동일한 효율성과 분해 성능이 나타날지는 추가적인 연구가 필요하다. 또한, rank $r$의 설정값이 데이터셋의 규모에 따라 성능에 영향을 미치는데, 이에 대한 최적의 선택 기준이 명확히 제시되지 않은 점이 아쉽다.

## 📌 TL;DR

본 논문은 의료 파운데이션 모델의 범용성과 특수성 사이의 트레이드오프를 해결하기 위해, 모델을 여러 개의 경량 전문가 모델로 나누는 **LoRKD (Low-Rank Knowledge Decomposition)** 프레임워크를 제안한다. 저차원 전문가 모듈과 EKS Convolution을 통해 계산 비용을 낮추면서도 각 도메인에 특화된 지식을 학습시켰으며, 실험을 통해 기존의 지식 분해 방법(KF) 및 MTL 방식보다 우수한 성능과 전이 가능성을 입증하였다. 이 연구는 향후 거대 의료 모델을 실제 병원 환경(과별 맞춤형 모델)에 효율적으로 배포하는 데 중요한 기여를 할 것으로 보인다.
