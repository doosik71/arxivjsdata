# Self-Supervised Prototypical Transfer Learning for Few-Shot Classification

Carlos Medina, Arnout Devos, Matthias Grossglauser (2020)

## 🧩 Problem to Solve

Few-shot classification은 학습 과정에서 보지 못한 새로운 클래스를 소수의 예제(shots)만을 가지고 분류기에 적응시켜 분류하는 학습 태스크이다. 기존 few-shot classification 접근법 대부분은 사전 학습(pre-training) 단계에서 목표 태스크 도메인과 관련된 비용이 많이 드는 주석(annotated) 데이터에 의존한다. 특히, 현실적인 도메인 시프트(domain shift) 상황에서 메타 학습(meta-learning) 방법론은 훈련 및 새로운 클래스가 다른 분포에서 오는 교차 도메인(cross-domain) 설정에서 일반적인 전이 학습(transfer learning)보다 성능이 떨어진다는 문제가 제기되었다.

이러한 배경에서, 논문의 목표는 레이블이 지정되지 않은 데이터만을 사용하여 사전 학습된 임베딩을 통해 few-shot classification 성능을 향상시키는 전이 학습 접근 방식을 제안하는 것이다. 이는 특히 레이블이 없는 훈련 도메인에서 self-supervised 방식으로 학습하고, 소수의 레이블이 있는 타겟 도메인 태스크로 효율적으로 전이될 수 있는 방법을 모색한다.

## ✨ Key Contributions

본 논문의 중심 아이디어는 self-supervised learning의 발전을 활용하여, 레이블이 지정되지 않은 원본 이미지와 그 증강(augmentation)된 샘플들을 임베딩 공간에서 가깝게 군집화하는 메트릭 임베딩을 학습하는 전이 학습 접근 방식인 ProtoTransfer를 제안하는 것이다. 이 사전 학습된 임베딩은 클래스 프로토타입(prototypes)을 요약하고 미세 조정(fine-tuning)을 통해 few-shot classification의 시작점으로 활용된다.

주요 기여 사항은 다음과 같다.

1. ProtoTransfer가 mini-ImageNet few-shot classification 태스크에서 최신 비지도 메타 학습(unsupervised meta-learning) 방법론들보다 4%에서 8% 높은 성능을 보이며, Omniglot에서도 경쟁력 있는 성능을 달성한다는 것을 입증한다.
2. 완전 지도 학습(fully supervised setting)과 비교하여, ProtoTransfer는 mini-ImageNet 및 CDFSL 벤치마크의 여러 교차 도메인 데이터셋에서 경쟁력 있는 성능을 달성하며, 훈련 과정에서 레이블을 요구하지 않는 이점을 제공한다.
3. 어블레이션 연구(ablation study) 및 교차 도메인 실험을 통해, 에피소드 메타 학습(episodical meta-learning)보다 더 많은 수의 동등한 훈련 클래스를 사용하고, 파라미터 미세 조정(parametric fine-tuning)이 지도 학습 방식과 유사한 성능을 얻는 데 핵심적인 역할을 한다는 것을 보여준다.

## 📎 Related Works

논문은 다음과 같은 관련 연구들과 차별점을 제시한다.

* **비지도 메타 학습 (Unsupervised meta-learning)**: CACTUs (Hsu et al., 2019)와 UFLST (Ji et al., 2019)는 클러스터링을 통해 서포트(support) 및 쿼리(query) 세트를 생성한 후 표준 메타 학습을 적용한다. UMTRA (Khodadadeh et al., 2019), AAL (Antoniou and Storkey, 2019), ULDA (Qin et al., 2020)는 이미지 증강을 통해 인공적인 few-shot 태스크를 생성한다.
  * **한계**: 이 방법론들은 하위 few-shot classification 태스크와 동일한 작은 인공 N-way K-shot 태스크를 생성해야 하는 메타 학습에 의존한다.
  * **차별점**: ProtoTransfer는 메타 학습을 사용하지 않고, 더 큰 배치 크기($N$)를 활용하여 더 강력한 임베딩을 학습한다. 또한, 하나의 증강되지 않은 서포트 샘플과 여러 개의 쿼리 샘플을 사용하여 더 나은 그래디언트 신호를 얻는다.
* **Self-supervision을 활용한 지도 메타 학습 (Supervised meta-learning aided by self-supervision)**: 일부 연구(Gidaris et al., 2019; Liu et al., 2019; Chen et al., 2019; Su et al., 2019)는 self-supervised 손실을 지도 메타 학습 에피소드와 함께 사용하거나 모델 초기화에 사용한다.
  * **차별점**: ProtoTransfer는 훈련 과정에서 어떠한 레이블도 요구하지 않는다.
* **Few-shot classification을 위한 미세 조정 (Fine-tuning for few-shot classification)**: Chen et al. (2019)은 타겟 태스크에 대한 적응(adaptation)이 교차 도메인 few-shot classification 성능에 중요하다고 제시한다. Triantafillou et al. (2020)은 지도 메타 학습 후 프로토타입으로 최종 레이어를 초기화하지만, 항상 모델의 모든 파라미터를 미세 조정한다.
  * **차별점**: ProtoTune은 유사하게 프로토타입 기반 초기화를 사용하지만, 임베딩 함수의 파라미터를 고정하고 최종 분류 레이어만 미세 조정할 수 있다.
* **대조 손실 학습 (Contrastive loss learning)**: 대조 손실은 강력한 임베딩 함수 학습에 큰 진전을 가져왔다(Ye et al., 2019; Chen et al., 2020; He et al., 2019; Tian et al., 2020; Li et al., 2020).
  * **차별점**: Ye et al. (2019)은 배치당 대조 손실을 제안하지만, ProtoTransfer와 달리 프로토타입당 여러 개의 증강된 쿼리 이미지를 사용하지 않고 2개의 추가 완전 연결 레이어를 사용한다. Li et al. (2020)도 프로토타입 기반 대조 손실을 사용하지만, k-Means 클러스터링을 통해 프로토타입을 계산하며 학습과 클러스터링 절차를 분리한다. ProtoTransfer는 이 두 과정을 단일 절차로 통합한다.

## 🛠️ Methodology

ProtoTransfer는 크게 두 단계로 구성된다: Self-Supervised Prototypical Pre-Training (ProtoCLR)과 Supervised Prototypical Fine-Tuning (ProtoTune). 이 두 단계는 few-shot classification을 위해 메트릭 임베딩을 학습하고 적응시키는 역할을 한다.

### 전체 파이프라인 및 시스템 구조

* **Self-Supervised Prototypical Pre-Training (ProtoCLR)**: 레이블이 없는 훈련 데이터셋 $D_b$를 사용하여 임베딩 함수 $f_{\theta}$를 학습한다. 이 단계에서는 각 이미지를 독립적인 클래스 프로토타입으로 간주하고, 해당 이미지의 증강된 버전들이 임베딩 공간에서 원본 이미지 주변에 군집되도록 대조 손실을 최소화한다.
* **Supervised Prototypical Fine-Tuning (ProtoTune)**: 사전 학습된 임베딩 함수 $f_{\theta}$를 고정한 채, 타겟 few-shot 태스크의 서포트 세트(support set) $S$에서 제공되는 소수의 레이블된 예제를 사용하여 최종 분류 레이어를 미세 조정한다. 클래스 프로토타입은 서포트 세트의 각 클래스 샘플 평균으로 계산되며, 이를 바탕으로 최종 레이어가 초기화된다.

### 2.2 Self-Supervised Prototypical Pre-Training: ProtoCLR

ProtoCLR은 모든 학습 단계를 $N$-way 1-shot classification 태스크로 구성하여 대조 손실 함수로 최적화한다.

**주요 구성 요소 및 역할:**

* **배치 생성 (Batch generation)**: 각 미니 배치($\{x_i\}_{i=1...N}$)는 훈련 세트에서 무작위로 추출된 $N$개의 샘플로 구성된다. 레이블 정보가 없으므로, 각 샘플 $x_i$는 자체적인 클래스로 간주되며, 1-shot 서포트 샘플이자 클래스 프로토타입 역할을 한다. 각 프로토타입 $x_i$에 대해 $Q$개의 무작위로 증강된 버전 $\tilde{x}_{i,q}$가 쿼리 샘플로 사용된다.

* **대조 프로토타입 손실 최적화 (Contrastive prototypical loss optimization)**: 이 손실은 증강된 쿼리 샘플 $\{\tilde{x}_{i,q}\}$이 임베딩 공간에서 해당하는 프로토타입 $x_i$ 주위에 군집되도록 유도한다.

**주요 방정식 설명:**

손실 함수 $L$은 다음과 같이 정의된다.

$$
\mathcal{L} = \frac{1}{NQ} \sum_{i=1}^{N} \sum_{q=1}^{Q} \mathcal{l}(i,q)
$$

여기서 $\mathcal{l}(i,q)$는 다음과 같은 소프트맥스(softmax) 교차 엔트로피 손실이다.

$$
\mathcal{l}(i,q) = -\log \frac{\exp(-d[f(\tilde{x}_{i,q}), f(x_i)])}{\sum_{k=1}^{N} \exp(-d[f(\tilde{x}_{i,q}), f(x_k)])}
$$

* $f(\cdot)$은 임베딩 함수 $f_{\theta}(\cdot)$를 나타낸다.
* $d[\cdot, \cdot]$는 두 임베딩 사이의 거리를 측정하는 함수이다. 본 논문에서는 유클리드 거리(Euclidean distance)를 사용한다.
* 분자($\exp(-d[f(\tilde{x}_{i,q}), f(x_i)])$)는 쿼리 샘플 $f(\tilde{x}_{i,q})$가 자신의 원본 프로토타입 $f(x_i)$에 얼마나 가까운지를 나타낸다. 거리가 짧을수록(즉, $d$ 값이 작을수록) 값이 커진다.
* 분모($\sum_{k=1}^{N} \exp(-d[f(\tilde{x}_{i,q}), f(x_k)])$)는 쿼리 샘플 $f(\tilde{x}_{i,q})$가 배치 내 모든 $N$개 프로토타입 $f(x_k)$에 대해 얼마나 가까운지를 나타내는 합이다.

이 손실 함수는 각 쿼리 샘플이 자신의 원본 이미지에 해당하는 프로토타입에 가까워지고, 배치 내 다른 이미지의 프로토타입으로부터는 멀어지도록 임베딩 공간을 학습시킨다. 이는 임베딩 파라미터 $\theta$에 대해 미니 배치 확률적 경사 하강법(SGD)을 사용하여 최소화된다.

### 2.3 Supervised Prototypical Fine-Tuning: ProtoTune

사전 학습된 임베딩 함수 $f_{\theta}(\cdot)$를 기반으로, few-shot classification의 타겟 태스크를 해결하기 위해 ProtoNet (Snell et al., 2017)의 프로토타입 기반 가장 가까운 이웃 분류기를 확장한다.

**훈련 목표 및 절차:**

1. **클래스 프로토타입 계산**: few-shot 태스크의 서포트 세트 $S$에 있는 각 클래스 $n$의 샘플들의 임베딩 평균을 계산하여 클래스 프로토타입 $c_n$을 생성한다.
    $$
    c_n = \frac{1}{|S_n|} \sum_{(x_i, y_i=n) \in S} f_{\theta}(x_i)
    $$
    여기서 $S_n$은 클래스 $n$에 속하는 서포트 샘플들의 집합이다.
2. **최종 선형 레이어 초기화**: ProtoNet의 파생을 따라, 최종 선형 분류 레이어의 가중치 $W_n$과 편향 $b_n$을 계산된 프로토타입 $c_n$으로 초기화한다.
    $$
    W_n = 2c_n
    $$
    $$
    b_n = -||c_n||^2
    $$
3. **미세 조정**: 임베딩 함수 파라미터 $\theta$는 고정된 채로, 이 최종 선형 레이어만 서포트 세트 $S$의 샘플들에 대해 소프트맥스 교차 엔트로피 손실을 사용하여 미세 조정된다. 이는 여러 예제가 클래스당 주어질 때 성능을 향상시키는 데 기여한다.

## 📊 Results

ProtoTransfer의 성능은 다양한 few-shot classification 시나리오에서 벤치마킹되고 분석되었다.

### 3.1 In-Domain Few-shot Classification: Omniglot and mini-ImageNet

* **데이터셋 및 작업**: Omniglot (28x28 흑백 이미지), mini-ImageNet (84x84 컬러 이미지)에서 $N$-way $K$-shot 분류 태스크.
* **기준선**: CACTUs, UMTRA, AAL, UFLST, ULDA 등 비지도 메타 학습 방법론과 MAML, ProtoNet (지도 메타 학습), Pre+Linear (지도 전이 학습).
* **지표**: 분류 정확도(Accuracy %).
* **설정**: Conv-4 아키텍처 사용, 배치 크기 $N=50$, 쿼리 증강 수 $Q=3$.
* **주요 정량적 결과 (Table 1)**:
  * **mini-ImageNet**: ProtoTransfer는 모든 최신 비지도 사전 학습 접근법을 4%에서 최대 8%까지 능가한다. 또한, 지도 메타 학습 방법인 MAML보다 우수한 성능을 보이는 경우가 많다. 이는 훈련 중 훨씬 적은 수의 레이블을 요구한다는 이점과 함께 나타난다.
  * **Omniglot**: ProtoTransfer는 대부분의 비지도 메타 학습 접근법과 경쟁력 있는 성능을 보여준다.

### 3.2 Cross-domain Few-Shot Classification: CDFSL benchmark

* **데이터셋 및 작업**: mini-ImageNet으로 훈련 후, CropDiseases, EuroSAT, ISIC2018, ChestX (mini-ImageNet과의 유사도 감소 순)와 같은 다양한 도메인의 데이터셋으로 교차 도메인 few-shot classification 태스크를 수행한다.
* **기준선**: ProtoNet, Pre+Mean-Centroid, Pre+Linear (지도 전이 학습), UMTRA-ProtoNet, UMTRA-ProtoTune (비지도 메타 학습).
* **설정**: ResNet-10 아키텍처 사용, 배치 크기 $N=50$, 쿼리 증강 수 $Q=3$. ProtoTransfer는 few-shot 미세 조정 단계에서 모델의 모든 파라미터가 미세 조정된다.
* **주요 정량적 결과 (Table 2)**:
  * ProtoTransfer는 메타 학습 기반 비지도 접근법들보다 최소 0.7%에서 최대 19%까지 일관되게 우수한 성능을 보인다.
  * 지도 전이 학습 방법론과 대부분 동등한 성능을 달성한다.
  * 특히 도메인 시프트가 가장 큰 ChestX 데이터셋에서는 ProtoTransfer가 모든 다른 접근법을 능가하는 결과를 보여준다.
  * 5-shot부터는 파라미터 미세 조정이 1%에서 13%까지 상당한 성능 향상을 제공한다 (UMTRA-ProtoNet vs. UMTRA-ProtoTune 비교).

### 3.3 Ablation Study: Batch Size, Number of Queries, and Fine-Tuning

* **설정**: mini-ImageNet, Conv-4 아키텍처.
* **변수**: 훈련 이미지 배치 크기 ($N$), 훈련 쿼리 수 ($Q$), 타겟 태스크 미세 조정 (FT).
* **주요 정량적 결과 (Table 3)**:
  * **배치 크기**: UMTRA-ProtoNet ($N=5$)에서 ProtoCLR-ProtoNet ($N=50$)으로 배치 크기를 늘리면 5%에서 9%의 성능 향상이 발생한다. 이는 self-supervised representation learning에 큰 배치 크기가 중요함을 시사한다.
  * **쿼리 수**: 훈련 쿼리 수를 $Q=3$으로 늘리면 미미하지만 일관된 성능 향상을 가져온다.
  * **미세 조정**: 많은 샷(shots)이 제공될 때 (예: 50-shot), 미세 조정은 상당한 성능 향상을 제공한다. ProtoTransfer는 미세 조정 전에도 few-shot 영역에서 경쟁력 있는 성능을 달성한다.

### 3.4 Number of Training Classes and Samples

* **설정**: mini-ImageNet (클래스 수 또는 샘플 수 감소), CUB (낮은 클래스 다양성).
* **기준선**: 지도 전이 학습 baseline Pre+Linear.
* **주요 정량적 결과 (Figure 2, Table 4)**:
  * **훈련 이미지 수 감소 (Figure 2a)**: 모든 클래스에서 균일하게 이미지 수를 줄일 때, few-shot classification 정확도는 감소하며 ProtoTransfer와 지도 baseline의 성능은 거의 일치한다.
  * **훈련 클래스 수 감소 (Figure 2b)**: 훈련 클래스 수가 16개 미만으로 떨어질 때 ProtoTransfer는 지도 baseline을 일관되고 현저하게 능가한다. 예를 들어 2개의 훈련 클래스를 가진 20-shot 경우, ProtoTransfer는 지도 baseline보다 16.9% 더 높은 성능을 보인다.
  * **교차 도메인 설정 (CUB $\rightarrow$ mini-ImageNet, Table 4)**: CUB 데이터셋 (조류 200클래스)으로 훈련하고 mini-ImageNet으로 테스트할 때, ProtoTransfer는 훈련 클래스의 다양성이 제한적일 때 지도 접근법보다 2%에서 4% 더 우수한 전이 정확도를 보인다. 이는 self-supervised 방식이 낮은 훈련 클래스 다양성에도 불구하고 판별적 특징(discriminative features)을 학습할 수 있음을 시사한다.

### 3.5 Task Generalization Gap

* **설정**: t-SNE (Maaten and Hinton, 2008)를 사용하여 ProtoCLR과 지도 학습 counterpart인 ProtoNet의 임베딩 공간 시각화 (Figure 3), mini-ImageNet의 훈련 및 테스트 스플릿에서 성능 비교 (Table 5).
* **주요 정성/정량적 결과 (Figure 3, Table 5)**:
  * **t-SNE 시각화 (Figure 3)**: 지도 ProtoNet은 훈련 클래스 내에서 더 많은 구조를 보여주지만, ProtoCLR은 훈련 및 테스트 클래스 모두에서 더 밀접하게 관련된 임베딩을 보여준다. 이는 ProtoCLR이 훈련 클래스와 테스트 클래스 간의 일반화 격차가 더 작을 수 있음을 시사한다.
  * **성능 비교 (Table 5)**: UMTRA 및 ProtoCLR과 같은 self-supervised 임베딩 접근법은 지도 ProtoNet보다 훨씬 작은 태스크 일반화 격차(task generalization gap)를 보인다. ProtoCLR은 분류 성능 저하가 거의 없는 반면, 지도 ProtoNet은 6%에서 12%의 상당한 정확도 감소를 겪는다.

## 🧠 Insights & Discussion

### 논문에서 뒷받침되는 강점

* **우수한 비지도 학습 성능**: ProtoTransfer는 mini-ImageNet에서 기존의 최신 비지도 메타 학습 방법론들을 크게 능가하며, 이는 레이블 없이도 강력한 표현(representation)을 학습할 수 있음을 입증한다.
* **레이블 효율성**: 지도 학습 방식과 비교하여 훨씬 적은 수의 레이블을 필요로 하면서도 경쟁력 있는 성능을 달성하여, 레이블링 비용이 높은 분야에서 실용적인 대안이 될 수 있다.
* **도메인 시프트에 대한 강건성**: 교차 도메인 설정인 CDFSL 벤치마크에서 지도 전이 학습 방법과 대등하거나 더 나은 성능을 보여주며, 특히 도메인 시프트가 큰 ChestX 데이터셋에서 우수성을 입증했다.
* **대규모 배치 및 미세 조정의 중요성**: 어블레이션 연구를 통해 대규모 배치 크기가 강력한 임베딩 학습에 결정적이며, 파라미터 미세 조정이 높은 샷(shot) 수에서 성능을 크게 향상시킬 수 있음을 명확히 보여준다.
* **낮은 클래스 다양성에 대한 강건성**: 훈련 클래스 수가 적거나 다양성이 낮은 환경(CUB 데이터셋)에서 지도 baseline보다 우수한 성능을 보여, self-supervised 접근 방식이 판별적 특징 학습에 효과적임을 시사한다.
* **작은 일반화 격차**: self-supervised 학습된 임베딩이 훈련 및 테스트 태스크 간의 성능 저하가 적어 일반화 능력이 우수함을 t-SNE 시각화와 정량적 결과로 뒷받침한다.

### 한계, 가정 또는 미해결 질문

* **계산 자원 (Computational Resources)**: 논문은 ProtoTransfer가 "orders of magnitude fewer labels"를 요구한다고 명시하지만, 사전 학습 과정의 계산 자원 요구량(예: 훈련 시간, GPU 메모리)에 대해서는 다른 방법론과 비교하여 명확하게 언급하지 않는다. 특히 큰 배치 사이즈($N=50$)를 사용하는 것이 계산적으로 더 부담스러울 수 있다.
* **하이퍼파라미터 튜닝**: 논문은 "limited hyperparameter tuning"을 수행했다고 언급하며, 이는 특정 데이터셋이나 아키텍처에 대한 최적의 하이퍼파라미터 탐색이 충분히 이루어지지 않았을 가능성을 내포한다. 실제 적용 시 추가적인 튜닝이 필요할 수 있다.
* **증강 전략의 일반화**: 각 데이터셋에 대해 증강 전략을 조정한다고 언급되어 있으나(Appendix A.3), 이러한 증강 전략이 새로운 도메인이나 태스크에 대해 얼마나 일반화될 수 있을지는 추가적인 연구가 필요하다.
* **단일 이미지 프로토타입의 한계**: ProtoCLR 단계에서 각 원본 이미지를 1-shot 프로토타입으로 간주하는 방식은 이상적인 경우(즉, 각 이미지가 하나의 명확한 개념을 대표하는 경우)에 효과적일 수 있으나, 더 복잡한 이미지나 다중 객체를 포함하는 이미지에서는 그 효율성이 떨어질 수 있다.
* **파라미터 고정 여부**: ProtoTune 단계에서 미세 조정 시 임베딩 함수의 파라미터를 고정하는 방식이 모든 시나리오에서 최적의 선택인지는 추가 분석이 필요하다. 특정 교차 도메인에서는 backbone을 함께 미세 조정한다(CDFSL 벤치마크). 어떤 경우에 어떤 파라미터를 미세 조정하는 것이 가장 효과적인지에 대한 심층적인 분석이 부족하다.

### 논문에 근거한 간략한 비판적 해석 및 논의사항

ProtoTransfer는 레이블이 없는 데이터로부터 효과적인 임베딩을 학습하고 이를 few-shot classification에 전이하는 강력한 프레임워크를 제시한다. 특히 비지도 메타 학습의 약점을 보완하고, 지도 학습 방식에 필적하는 성능을 레이블 효율적으로 달성한다는 점에서 중요한 기여를 한다.

그러나 이러한 성과는 주로 대규모 배치 크기, 세심하게 설계된 증강 전략, 그리고 특정 시나리오에서의 파라미터 미세 조정 덕분이다. 이는 self-supervised learning 분야에서 대조 학습(contrastive learning)이 강력한 표현 학습에 필수적이라는 최근의 추세와 일치한다. 낮은 클래스 다양성 환경에서의 우수한 성능은 기존 지도 학습이 클래스 구분에 너무 의존하여 일반화에 실패하는 지점을 self-supervision이 우회할 수 있음을 보여주는 흥미로운 통찰이다.

미해결 질문으로는 대규모 배치 학습의 계산 비용과 다양한 데이터셋에 대한 증강 전략의 자동화 또는 일반화 가능성 등이 있다. 향후 연구는 이러한 실용적인 측면을 개선하고, self-supervised pre-training이 왜 특정 상황(예: 낮은 클래스 다양성)에서 지도 학습보다 더 효과적인지에 대한 이론적 이해를 심화하는 데 집중할 수 있다. 전반적으로 ProtoTransfer는 few-shot learning 분야에서 비지도 학습의 잠재력을 크게 확장하며, 실제 적용 가능성을 높이는 중요한 진전을 이룬다.

## 📌 TL;DR

ProtoTransfer는 레이블이 없는 데이터를 활용하는 self-supervised 방식으로 few-shot classification 문제를 해결한다. 핵심 아이디어는 원본 이미지와 그 증강된 버전들을 임베딩 공간에서 가깝게 군집화하도록 학습하는 'ProtoCLR'이라는 사전 학습 단계와, 사전 학습된 임베딩을 기반으로 타겟 태스크에 맞게 최종 분류 레이어를 미세 조정하는 'ProtoTune' 단계로 구성된다.

이 연구의 주요 기여는 다음과 같다.

1. 기존 비지도 메타 학습 방법들보다 mini-ImageNet에서 훨씬 높은 성능을 달성한다.
2. 교차 도메인 시나리오에서 지도 학습 방식과 경쟁력 있는 성능을 보이면서도 훈련에 레이블이 필요 없어 레이블링 비용을 크게 절감한다.
3. 대규모 배치 학습과 파라미터 미세 조정이 성능 향상에 결정적임을 입증한다.
4. 훈련 클래스의 다양성이 낮더라도 지도 학습 baseline보다 뛰어난 일반화 능력을 보인다.

이 연구는 self-supervised learning이 few-shot classification에서 강력한 임베딩을 학습하는 효과적인 방법임을 보여주며, 특히 레이블이 부족한 의료 영상이나 로봇 공학 같은 실제 적용 분야에서 향후 연구 및 활용에 중요한 역할을 할 가능성이 크다.
