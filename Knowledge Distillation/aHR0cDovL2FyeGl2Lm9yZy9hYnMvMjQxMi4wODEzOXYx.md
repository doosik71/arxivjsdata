# Wasserstein Distance Rivals Kullback-Leibler Divergence for Knowledge Distillation

Jiaming Lv, Haoyuan Yang, Peihua Li (2024)

## 🧩 Problem to Solve

본 논문은 지식 증류(Knowledge Distillation, KD) 분야에서 오랫동안 표준으로 사용되어 온 Kullback-Leibler Divergence (KL-Div)의 근본적인 한계를 해결하고자 한다. 저자들은 KL-Div가 가진 두 가지 주요 문제점을 지적한다.

첫째, 로짓 증류(Logit Distillation) 측면에서 KL-Div는 교사(Teacher) 모델과 학생(Student) 모델 간의 동일한 카테고리에 대한 확률만을 직접적으로 비교하는 'category-to-category' 방식이다. 이로 인해 실제 세계의 데이터가 가진 클래스 간의 복잡한 상호관계(Inter-category Relations, IRs), 즉 특정 클래스들이 서로 얼마나 유사하거나 다른지에 대한 풍부한 정보를 명시적으로 활용하지 못한다.

둘째, 중간 계층의 특징 증류(Feature Distillation) 측면에서 KL-Div는 고차원 공간에서 특징들이 매우 희소하게 분포하는 특성으로 인해 비모수적 밀도 추정이 어렵고, 분포가 서로 겹치지 않는 경우(non-overlapping distributions) 적절한 거리를 측정하지 못한다. 또한, KL-Div는 거리 척도(metric)가 아니며 하부 매니폴드의 기하학적 구조를 인식하지 못한다는 단점이 있다.

따라서 본 논문의 목표는 Wasserstein Distance (WD)를 도입하여 클래스 간 상호관계를 활용하는 로짓 증류 방법(WKD-L)과 기하학적 구조를 반영하는 특징 증류 방법(WKD-F)을 제안함으로써 기존 KL-Div 기반의 KD 성능을 뛰어넘는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 전송 비용(transport cost)의 개념을 가진 Wasserstein Distance를 KD에 적용하여, 단순한 점별(point-wise) 비교를 넘어선 분포 간의 최적 운송(Optimal Transport) 관점에서 지식을 전달하는 것이다.

1. **WKD-L (Logit Distillation):** 이산적 WD(Discrete WD)를 사용하여 예측 확률 분포를 일치시킨다. 특히 Centered Kernel Alignment (CKA)를 통해 클래스 간의 상호관계를 정량화하고, 이를 WD의 전송 비용으로 설정함으로써 클래스 간의 의미적 유사성을 증류 과정에 명시적으로 반영한다.
2. **WKD-F (Feature Distillation):** 연속적 WD(Continuous WD)를 사용하여 중간 계층의 특징 분포를 일치시킨다. 특징 분포를 가우시안(Gaussian) 분포로 모델링하고, 가우시안 공간에서의 리만 메트릭(Riemannian metric)인 WD를 통해 특징의 기하학적 구조를 보존하며 지식을 전달한다.

## 📎 Related Works

기존의 KD 연구들은 주로 다음과 같은 방향으로 진행되었다.

- **KL-Div 기반 방법:** Hinton 등의 고전적 KD를 시작으로, 최근에는 타겟 클래스와 비타겟 클래스를 분리하여 학습하는 DKD, NKD, WTTM 등이 제안되었다. 이들은 성능 향상을 이루었으나, 여전히 클래스 간 상호관계를 명시적으로 활용하지 못하는 한계가 있다.
- **WD 기반 방법:** 일부 연구(WCoRD, EMD+IPOT 등)가 WD를 도입하였으나, 이들은 주로 미니 배치 내의 인스턴스 간 매칭(cross-instance matching)에 집중하였다. 결과적으로 클래스 간의 관계를 활용하는 로짓 증류보다는 특징 증류에 치중되어 있으며, 비모수적 방법을 사용하여 성능이 KL-Div 기반 최신 방법들에 비해 낮았다.
- **통계적 모델링 기반 방법:** NST나 ICKD-C 등은 특징의 1차, 2차 모멘트를 활용하여 분포를 일치시키려 했으나, 이는 주로 Frobenius norm에 기반한 것으로, 대칭 양정치(SPD) 행렬이 이루는 리만 공간의 기하학적 특성을 무시했다는 한계가 있다.

## 🛠️ Methodology

### 1. Discrete WD for Logit Distillation (WKD-L)

WKD-L은 교사와 학생 모델의 로짓 확률 분포 간의 거리를 최소화한다.

**카테고리 상호관계(IRs) 정량화:**
먼저 클래스 간의 유사도를 측정하기 위해 CKA를 사용한다. 각 클래스 $C_i$에 대해 특징 행렬 $X_i$를 생성하고 커널 행렬 $K_i$를 계산하여 다음과 같이 IR을 정의한다.
$$IR(C_i, C_j) = \frac{HSIC(C_i, C_j)}{\sqrt{HSIC(C_i, C_i)}\sqrt{HSIC(C_j, C_j)}}$$
여기서 $HSIC$는 Hilbert-Schmidt Independence Criterion이며, 이를 통해 클래스 간의 통계적 유사성을 $[0, 1]$ 범위로 측정한다.

**손실 함수:**
이산적 WD는 한 분포를 다른 분포로 변환하는 최소 비용으로 정의된다. 전송 비용 $c_{ij}$는 위에서 구한 $IR$을 가우시안 커널을 통해 거리로 변환하여 설정한다.
$$c_{ij} = 1 - \exp(-\kappa(1 - IR^T(C_i, C_j)))$$
최종적인 WKD-L의 손실 함수는 다음과 같이 타겟 클래스와 비타겟 클래스를 분리하여 정의한다.
$$L_{WKD\text{-}L} = \lambda D_{WD}(p^T_{\setminus t}, p^S_{\setminus t}) + L_t$$
여기서 $D_{WD}$는 엔트로피 정규화된 선형 계획법(entropy regularized linear programming)을 통해 계산되는 이산적 WD이며, $L_t$는 타겟 클래스에 대한 교차 엔트로피 손실이다.

### 2. Continuous WD for Feature Distillation (WKD-F)

WKD-F는 중간 계층 특징의 분포를 가우시안 분포로 모델링하여 전달한다.

**특징 분포 모델링:**
입력 이미지의 특징 맵 $F \in \mathbb{R}^{l \times m}$에 대해 평균 $\mu$와 공분산 $\Sigma$를 계산하여 가우시안 분포 $\mathcal{N}(\mu, \Sigma)$로 모델링한다.
$$\mu = \frac{1}{m} \sum_{i=1}^{m} f_i, \quad \Sigma = \frac{1}{m} \sum_{i=1}^{m} (f_i - \mu)(f_i - \mu)^T$$

**손실 함수:**
두 가우시안 분포 $\mathcal{N}^T$와 $\mathcal{N}^S$ 사이의 연속적 WD는 다음과 같은 닫힌 형태(closed form)로 계산된다.
$$D_{WD}(\mathcal{N}^T, \mathcal{N}^S) = D_{mean}(\mu^T, \mu^S) + D_{cov}(\Sigma^T, \Sigma^S)$$
$$D_{mean}(\mu^T, \mu^S) = \|\mu^T - \mu^S\|_2$$
$$D_{cov}(\Sigma^T, \Sigma^S) = \text{tr}(\Sigma^T + \Sigma^S - 2((\Sigma^T)^{1/2} \Sigma^S (\Sigma^T)^{1/2})^{1/2})$$
최종 손실 함수는 평균과 공분산의 비중을 조절하는 $\gamma$를 도입하여 다음과 같이 정의한다.
$$L_{WKD\text{-}F} = \gamma D_{mean}(\mu^T, \mu^S) + D_{cov}(\Sigma^T, \Sigma^S)$$
실험적으로는 연산 효율성과 강건함을 위해 풀 공분산 행렬 대신 대각 공분산 행렬(Diagonal Covariance)을 사용하며, 이 경우 $D_{cov}$는 표준 편차 벡터 간의 $L_2$ 거리로 단순화된다.

## 📊 Results

### 실험 설정

- **데이터셋:** ImageNet, CIFAR-100, MS-COCO.
- **작업:** 이미지 분류(Image Classification), 객체 검출(Object Detection), 자기 지식 증류(Self-KD).
- **지표:** Top-1 Accuracy, mAP.
- **기준선:** KL-Div 기반(DKD, NKD, WTTM), 특징 기반(FitNet, CRD, ReviewKD).

### 주요 결과

1. **이미지 분류 (ImageNet/CIFAR-100):**
    - WKD-L은 기존의 강력한 KL-Div 변형 모델들보다 우수한 성능을 보였다. 특히 ResNet34 $\to$ ResNet18 설정에서 Top-1 정확도를 향상시켰다.
    - WKD-F는 FitNet, CRD 등 기존 특징 증류 방법들을 크게 상회하였으며, 특히 CNN과 Transformer 간의 이기종(heterogeneous) 구조 증류에서 매우 강력한 성능을 보였다.
    - WKD-L과 WKD-F를 결합했을 때 가장 높은 성능을 기록하였다.

2. **객체 검출 (MS-COCO):**
    - Faster R-CNN 프레임워크에 적용한 결과, WKD-L과 WKD-F 모두 기존 방법들보다 높은 mAP를 달성하였다. 특히 WKD-F는 이전 최고 성능 모델인 ReviewKD를 유의미하게 앞질렀다.

3. **자기 지식 증류 (Self-KD):**
    - BAN(Born-Again Network) 프레임워크에서 WKD-L을 적용한 결과, 기존 KL-Div 기반의 BAN보다 약 0.9% 높은 정확도를 달성하여 일반화 능력을 입증하였다.

### 정량적 분석 및 효율성

- **지연 시간(Latency):** WKD-L은 Sinkhorn 알고리즘으로 인해 KL-Div보다 약 1.3배 느리지만, WKD-F는 단순한 평균/분산 계산만 필요하므로 KL-Div와 유사한 속도를 보이며 ReviewKD보다 약 1.6배 빨랐다.

## 🧠 Insights & Discussion

**강점:**
본 연구는 KL-Div의 'point-wise' 비교 방식에서 벗어나 'distribution-wise' 비교 방식으로 전환함으로써 KD의 성능을 한 단계 끌어올렸다. 특히 로짓 단계에서는 클래스 간 상호관계를 이용해 "개"가 "자동차"보다 "늑대"와 더 유사하다는 정보를 학습에 활용했고, 특징 단계에서는 리만 메트릭을 통해 특징 공간의 기하학적 구조를 보존했다는 점이 매우 고무적이다.

**한계 및 비판적 해석:**

- **계산 비용:** WKD-L의 경우 최적 운송 문제를 풀어야 하므로 계산 비용이 증가한다. 비록 GPU 병렬 연산으로 보완 가능하지만, 실시간 학습이 중요한 환경에서는 부담이 될 수 있다.
- **분포 가정:** WKD-F는 특징 분포가 가우시안 분포를 따른다는 가정을 전제로 한다. 하지만 실제 딥러닝 모델의 고차원 특징이 반드시 가우시안 분포를 따르는지에 대해서는 명확한 근거가 부족하며, 이는 성능 최적화의 한계점으로 작용할 수 있다.
- **하이퍼파라미터 민감도:** $\lambda, \gamma, \kappa, \tau$ 등 조정해야 할 하이퍼파라미터가 많아, 새로운 데이터셋이나 모델에 적용할 때 튜닝 비용이 발생할 가능성이 크다.

## 📌 TL;DR

본 논문은 기존 지식 증류의 주류였던 KL-Divergence의 한계를 지적하고, 이를 대체할 **Wasserstein Distance (WD) 기반의 증류 프레임워크(WKD)**를 제안한다. **WKD-L**은 CKA를 통해 클래스 간 상호관계를 로짓 증류에 반영하고, **WKD-F**는 가우시안 분포 모델링과 연속적 WD를 통해 특징의 기하학적 구조를 증류한다. 실험 결과, 이미지 분류 및 객체 검출 등 다양한 태스크에서 기존 KL-Div 기반 및 최신 특징 증류 방법들을 압도하는 성능을 보였으며, 이는 WD가 복잡한 데이터 분포를 전달하는 데 훨씬 효율적인 척도임을 시사한다.
