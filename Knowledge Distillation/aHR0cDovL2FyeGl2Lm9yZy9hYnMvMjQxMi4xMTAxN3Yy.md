# On Distilling the Displacement Knowledge for Few-Shot Class-Incremental Learning

Pengfei Fang, Yongchun Qin, Hui Xue (2024)

## 🧩 Problem to Solve

본 논문은 Few-shot Class-Incremental Learning (FSCIL)에서 발생하는 두 가지 핵심 문제를 해결하고자 한다. 첫째는 새로운 클래스를 학습할 때 기존에 학습한 지식을 잃어버리는 Catastrophic Forgetting(치명적 망각) 현상이다. 둘째는 FSCIL의 특성상 새로운 클래스(Novel classes)의 데이터가 매우 적기 때문에, 판별력 있는 특징 표현(Discriminative feature representation)을 생성하기 어렵다는 점이다.

기존의 Knowledge Distillation (KD) 방식은 주로 개별 샘플의 출력값(Logits)을 보존하는 Individual Knowledge Distillation (IKD)이나 샘플 간의 유사도를 보존하는 Relational Knowledge Distillation (RKD)를 사용한다. 그러나 IKD는 데이터가 부족한 few-shot 상황에서 성능이 저하되며, RKD는 차원 축소 과정에서 방향성 정보를 소실하고 이상치(Outlier)에 민감하게 반응하여 강건성(Robustness)이 떨어진다는 한계가 있다. 따라서 본 논문의 목표는 샘플 간의 구조적 정보를 효율적으로 보존하면서도 이상치에 강건한 새로운 증류 기법을 제안하여 FSCIL의 성능을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 샘플 간의 단순한 유사도가 아닌, 특징 공간에서의 '변위(Displacement)'를 보존하는 것이다. 주요 기여 사항은 다음과 같다.

1.  **Displacement Knowledge Distillation (DKD) 제안**: 두 샘플 간의 차이(Point-wise difference)를 변위 벡터로 정의하고, 이를 확률 분포로 변환하여 선생(Teacher) 모델과 학생(Student) 모델 간의 분포 유사도를 유지한다. 이를 통해 거리와 방향 정보를 모두 보존하며 정보 밀도를 높인다.
2.  **Dual Distillation Network (DDNet) 설계**: 데이터 양에 따른 특징 분포의 차이를 고려하여, 풍부한 데이터를 가진 Base classes에는 IKD를 적용하고, 데이터가 적은 Novel classes에는 DKD를 적용하는 이원화된 증류 구조를 제안한다.
3.  **Instance-aware Sample Selector 도입**: 추론 단계에서 입력 샘플이 Base 클래스인지 Novel 클래스인지 동적으로 판단하여 두 브랜치의 가중치를 조절하는 샘플 선택기를 구현함으로써 각 방식의 상보적 강점을 활용한다.

## 📎 Related Works

FSCIL은 제한된 데이터로 새로운 클래스를 학습하는 Few-shot Learning (FSL)과 지속적으로 새로운 작업을 배우는 Class-incremental Learning (CIL)이 결합된 형태이다. 기존 접근 방식은 크게 세 가지로 나뉜다.

-   **Data-based methods**: 메모리 셋을 유지하여 데이터를 재사용하는 방식(Data rehearsal)이나 유사 샘플을 생성하는 방식이다.
-   **Ensemble-based methods**: 서로 다른 성질을 가진 모델들을 결합하여 안정성과 가소성(Stability-Plasticity) 사이의 균형을 맞추는 방식이다.
-   **Regularization-based methods**: 파라미터의 변화를 제한하거나 프로토타입 간의 위상(Topology)을 보존하여 망각을 방지하는 방식이다.

본 논문은 특히 Relational Distillation Knowledge (RKD)와 같은 구조적 증류 방식의 한계를 지적한다. RKD는 유사도 행렬을 통해 구조를 모델링하지만, 이는 저차원 메트릭으로 변환되는 과정에서 정보 손실이 발생하며, 샘플 간의 강한 결합으로 인해 이상치 하나가 전체 그래디언트에 큰 영향을 주는 취약점이 있다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 파이프라인
DDNet은 두 단계로 작동한다. 첫 단계에서는 Base classes 데이터를 사용하여 모델을 충분히 사전 학습시킨 후 이 모델을 고정(Frozen)하여 Base model로 저장한다. 두 번째 단계(Incremental stage)에서는 새로운 클래스가 들어올 때마다 현재 모델을 미세 조정하며, 이때 Base knowledge 보존을 위한 IKD와 Novel knowledge 보존을 위한 DKD를 동시에 적용한다.

### 2. Displacement Knowledge Distillation (DKD)
DKD는 샘플 쌍의 변위 벡터를 이용하여 구조적 정보를 추출한다. 두 샘플 $x_i$와 $x_j$의 로짓(Logit)을 각각 $z_i, z_j$라고 할 때, 이들의 차이인 $z_i - z_j$를 변위 벡터로 정의한다.

DKD의 손실 함수는 다음과 같이 정의된다:
$$L_{DKD} = \mathbb{E}_{(x,y) \sim D_m \cap D_p} \text{DKD}(z_f^\tau || z_f^{\tau-1})$$
$$\text{DKD}(z_f^\tau || z_f^{\tau-1}) = \frac{1}{N-1} \sum_{j \neq i} \text{KD}(z_f^\tau{_i} - z_f^\tau{_j} \| z_f^{\tau-1}{_i} - z_f^{\tau-1}{_j})$$

여기서 $\text{KD}(\cdot)$는 일반적으로 KL-Divergence를 의미하며, 변위 벡터를 Softmax를 통해 확률 분포로 변환하여 비교한다. 이 방식은 변위 벡터가 원래의 차원 공간 $\mathbb{R}^d$를 그대로 유지하므로 정보 손실이 없으며, 각 샘플 쌍을 독립적인 선생-학생 쌍으로 취급하여 샘플 간의 과도한 결합을 방지한다.

### 3. Dual Distillation Network (DDNet) 및 학습 절차
DDNet은 Base 클래스와 Novel 클래스에 서로 다른 전략을 적용한다.
-   **Base Knowledge 보존 (IKD)**: Base 클래스는 이미 충분히 학습되었으므로 로짓 값을 직접 보존하는 IKD를 사용한다.
    $$L_{IKD} = \mathbb{E}_{(x,y) \sim D_m \cap D_0} \text{KD}(z_f^\tau || z_f^0)$$
-   **Novel Knowledge 보존 (DKD)**: 데이터가 부족한 Novel 클래스는 샘플 간의 관계를 보존하는 DKD를 사용하여 특징 공간의 임베딩을 개선한다.

전체 특징 추출기 $f_\phi^\tau$의 최종 손실 함수는 분류 손실($L_{cls}$)과 증류 손실($L_{kd}$)의 합으로 구성된다:
$$L_f = L_{cls} + (w_1 \cdot L_{IKD} + w_2 \cdot L_{DKD})$$

### 4. Sample Selector
추론 시 Base 모델($f_\phi^0$)과 현재 모델($f_\phi^\tau$)의 출력을 적절히 혼합하기 위해 샘플 선택기 $g_\psi^\tau(x)$를 사용한다. 최종 예측값 $z_{pred}$는 다음과 같이 결정된다:
$$z_{pred} = z_g^\tau(0) \cdot z_f^\tau + z_g^\tau(1) \cdot z_f^0$$

샘플 선택기는 Base와 Novel 클래스를 구분하는 이진 분류기로, 모멘텀(Momentum) 기반의 프로토타입 업데이트와 Triplet Loss($L_{trip}$), Binary Cross-Entropy Loss($L_{bincls}$)를 사용하여 학습된다. 이를 통해 Base 클래스와 Novel 클래스 간의 결정 경계를 명확히 한다.

## 📊 Results

### 1. 실험 설정
-   **데이터셋**: CIFAR-100, miniImageNet, CUB-200.
-   **평가 지표**: Knowledge Retention (KR, $\text{Acc}_\tau / \text{Acc}_0 \times 100\%$) 및 Accuracy Drop (AD, $\text{Acc}_0 - \text{Acc}_\tau$).
-   **비교 대상**: TOPIC, EEIL, FACT, ALFSCIL, BiDist 등 최신 FSCIL 방법론들.

### 2. 주요 결과
-   **정량적 성능**: DDNet은 세 가지 벤치마크 모두에서 SOTA(State-of-the-art) 성능을 달성하였다. 특히 KR 값에서 기존 방법들보다 평균 2.62%의 성능 향상을 보였으며, 데이터 리플레이 방식인 BiDist와 비교했을 때도 유의미한 우위를 점했다.
-   **강건성 분석**: 훈련 세트에 1%에서 20%까지 이상치(Outlier)를 섞어 공격했을 때, DDNet은 BiDist보다 훨씬 적은 성능 하락(Accuracy Drop 감소)을 보였다. 이는 DKD가 유사도 기반의 결합을 피함으로써 이상치의 영향을 최소화했음을 입증한다.
-   **일반화 능력**: FSCIL 외에 일반적인 CIL 작업(LwF, iCaRL, WA)에 DKD를 적용했을 때도 평균 정확도가 상승하고 망각률(Average Forgetting)이 감소하는 결과를 얻었다. 또한 모델 압축(Model Compression) 작업에서도 기존 RKD보다 우수한 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 FSCIL에서 Base 클래스와 Novel 클래스의 특징 분포가 서로 다르다는 점에 주목하여, 이에 맞는 서로 다른 증류 전략을 적용해야 한다는 통찰을 제시하였다. 

**DKD의 이론적 강점**은 다음과 같다.
첫째, RKD가 샘플 간 관계를 $1$차원 유사도 값으로 축소하여 정보 손실이 발생하는 반면, DKD는 원본 차원 공간에서의 변위를 그대로 유지하여 방향성과 거리 정보를 모두 보존한다.
둘째, 그래디언트 분석 결과, RKD는 한 샘플의 이상치가 모든 샘플 쌍의 유사도에 영향을 주어 전체 그래디언트를 오염시키지만, DKD는 영향 범위가 제한적이어서 훨씬 강건하다.

다만, 샘플 선택기(Sample Selector)의 정확도가 최종 성능에 영향을 미치는데, Base와 Novel 클래스 간에 내재적인 의미론적 유사성이 부족하여 클러스터링이 완벽하지 않을 수 있다는 점이 한계로 언급된다.

## 📌 TL;DR

본 연구는 Few-shot Class-Incremental Learning에서 발생하는 망각 문제와 데이터 부족으로 인한 특징 표현 저하 문제를 해결하기 위해 **Displacement Knowledge Distillation (DKD)**와 **Dual Distillation Network (DDNet)**를 제안하였다. DKD는 샘플 간의 변위 벡터를 통해 구조적 정보를 보존하며 이상치에 강건한 특성을 가진다. DDNet은 Base 클래스에는 IKD를, Novel 클래스에는 DKD를 적용하고 샘플 선택기로 이를 통합하여 SOTA 성능을 달성하였다. 이 기법은 FSCIL뿐만 아니라 일반적인 CIL 및 모델 압축 작업에도 적용 가능한 범용적인 증류 방법론으로서의 잠재력을 보여주었다.