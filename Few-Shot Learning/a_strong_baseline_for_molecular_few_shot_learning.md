# A Strong Baseline for Molecular Few-Shot Learning

Philippe Formont, Hugo Jeannin, Pablo Piantanida, Ismail Ben Ayed (2025)

## 🧩 Problem to Solve

본 논문은 신약 개발 과정에서 필수적인 QSAR(Quantitative Structure-Activity Relationship, 정량적 구조-활성 관계) 모델링의 데이터 부족 문제를 해결하고자 한다. 새로운 약물 타겟에 대해 생물학적 활성을 예측하기 위해서는 실험 데이터가 필요하지만, 실험 비용이 매우 높고 시간이 오래 걸리기 때문에 가용한 데이터가 매우 적은 'Few-Shot' 상황이 빈번하게 발생한다.

기존의 분자 Few-Shot Learning(FSL) 연구들은 주로 복잡한 Meta-learning 전략에 의존해 왔다. 하지만 Meta-learning은 모델이 '학습하는 법을 학습'하도록 하는 특수한 전처리와 훈련 절차가 필요하며, 이는 모델 가중치에 접근할 수 없는 Black-box 설정(API를 통해서만 예측값을 얻는 경우)에서 적용하기 어렵다는 한계가 있다. 또한, 기존의 단순한 Fine-tuning 방식은 분자 데이터셋에서 Meta-learning보다 성능이 떨어진다는 보고가 있어, 효율적이고 강력한 단순 Fine-tuning 기반의 baseline을 구축하는 것이 본 논문의 목표이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 단순한 Linear probing을 넘어, 클래스별 공분산(Covariance)을 고려하는 **Quadratic Probe**를 제안하는 것이다.

1. **Mahalanobis Distance 기반의 Quadratic Probe 설계**: 단순히 코사인 유사도를 사용하는 대신, Mahalanobis 거리를 이용하여 쿼리 포인트와 클래스 프로토타입 간의 관계를 더 정교하게 측정한다.
2. **Degenerate Solution 방지를 위한 최적화 기법**: Quadratic Probe를 단순 Gradient Descent로 학습할 경우, 클래스가 선형 분리 가능할 때 가중치의 Frobenius norm이 무한히 커지는 퇴화(Degenerate) 문제가 발생함을 수학적으로 증명하고, 이를 해결하기 위해 **Block-coordinate descent**와 정규화된 손실 함수를 제안하였다.
3. **강력한 성능 및 강건성 입증**: 제안 방법이 기존의 최첨단 Meta-learning 방법들과 경쟁 가능한 수준의 성능을 보이며, 특히 데이터 분포가 변화하는 Domain shift 상황(클래스 불균형 등)에서 Meta-learning보다 훨씬 강건함을 입증하였다.
4. **새로운 벤치마크 도입**: Domain shift에 대한 모델의 강건성을 평가하기 위해 DTI(Drug-Target-Interaction) 및 Library Screening 데이터셋을 이용한 평가 체계를 구축하였다.

## 📎 Related Works

분자 Few-Shot Learning 분야에서는 주로 다음과 같은 접근 방식들이 사용되어 왔다.

- **Meta-learning**: Protonet, MAML, PAR, ADKT-IFT 등이 있으며, 이들은 서포트 세트(Support set)를 통해 빠르게 적응하도록 사전 훈련된다. 하지만 앞서 언급했듯 특수한 훈련 파이프라인이 필요하며 Black-box 설정에서 사용이 제한적이다.
- **Fine-tuning**: 일반적인 지도 학습으로 사전 훈련된 모델을 서포트 세트로 미세 조정하는 방식이다. Stanley et al. (2021)의 연구에서는 GNN-MT나 MAT 같은 Fine-tuning 방식이 Meta-learning보다 성능이 낮다고 보고하였다.
- **Foundational Models**: 최근 대규모 데이터로 사전 훈련된 기초 모델들이 등장하고 있으나, 이들이 실제 Few-Shot 분류 작업에서 얼마나 효율적으로 적응하는지에 대한 분석은 부족한 실정이다.

본 논문은 기존 Fine-tuning 방식이 성능이 낮았던 이유가 단순한 Linear probe에 의존했기 때문이라고 보고, 이를 Quadratic probe로 확장함으로써 Meta-learning의 복잡성 없이도 유사한 성능을 낼 수 있음을 보여주며 차별점을 둔다.

## 🛠️ Methodology

### 1. Model Pre-training

본 연구에서는 Principal Neighborhood Aggregation (PNA) 모델을 기반으로 한 GNN backbone $f_{\theta}$를 사용한다. 분자 그래프와 핑거프린트를 입력으로 받아 임베딩을 생성하며, 사전 훈련 단계에서는 FS-mol 벤치마크의 훈련 세트를 사용하여 Multitask 학습을 수행한다. 최종 임베딩 $z_i$는 다음과 같이 정규화된 형태로 추출된다.

$$z_i = \frac{f_{\theta}(x_i)}{\|f_{\theta}(x_i)\|}$$

### 2. Multitask Linear Probing

비교 대상이 되는 Linear probe는 각 클래스 $k$에 대해 단위 벡터 $w_k$를 학습하며, 다음과 같은 Softmax 확률 분포를 사용한다.

$$p_{i,k} = \frac{\exp(\tau \langle z_i, w_k \rangle + b_k)}{\sum_{k' \in C} \exp(\tau \langle z_i, w_{k'} \rangle + b_{k'})}$$

### 3. Quadratic Probing

본 논문의 핵심 제안으로, 코사인 유사도 대신 Mahalanobis 거리를 사용하여 클래스 프로토타입 $w_k$와 쿼리 $z_i$ 사이의 거리를 측정한다.

$$\|z_i - w_k\|^2_{M_k} = (z_i - w_k)^T M_k (z_i - w_k)$$

여기서 $M_k$는 정밀도 행렬(Precision matrix, $\Sigma_k^{-1}$)이며, 양의 준정부호(PSD) 행렬이다. 예측 확률은 다음과 같다.

$$p_{i,k} \triangleq \frac{\exp(-\|z_i - w_k\|^2_{M_k})}{\sum_{k' \in C} \exp(-\|z_i - w_{k'}\|^2_{M_{k'}})}$$

### 4. Optimization and Modified Loss

단순 Cross-entropy loss를 $M_k$에 대해 최적화하면, $\|M_k\|_F \to \infty$로 발산하며 모델이 과적합되는 퇴화 솔루션(Degenerate solution)이 발생한다. 이를 방지하기 위해 저자들은 손실 함수를 두 부분 $f_1, f_2$로 나누고, $f_2$를 다음과 같은 $\tilde{f}_2$로 대체한 수정된 손실 함수를 제안한다.

$$\tilde{f}_2(\Theta) \triangleq -\frac{\sum_{k \in \{0,1\}} |S_k| \log \det(M_k)}{|S|}$$

이 수정된 손실 함수는 $M_k$에 대해 다음과 같은 닫힌 형태의 해(Closed-form solution)를 가진다.

$$M_k = \left( \frac{1}{|S_y|} \sum_{i \in S_y} (z_i - w_k)(z_i - w_k)^T \right)^{-1}$$

최종 학습은 **Block-coordinate descent** 방식을 따른다.

1. Cross-entropy loss를 사용하여 $w_k$를 Gradient Descent로 한 단계 업데이트한다.
2. 위에서 도출된 닫힌 형태의 해를 사용하여 $M_k$를 즉시 업데이트한다.
3. 이때 수치적 안정성을 위해 Shrinkage 파라미터 $\lambda$를 도입하여 $\Sigma_k \triangleq (1-\lambda)\Sigma_{k, \text{empirical}} + \lambda I$로 계산한다.

## 📊 Results

### 1. FS-mol Benchmark

- **설정**: 5,000개의 타겟에 대해 서포트 세트 크기 $|S| \in \{16, 32, 64, 128\}$에서 성능을 측정하였다. 지표로는 $\Delta$AUCPR을 사용하였다.
- **결과**:
  - Quadratic Probe는 Linear Probe보다 일관되게 우수한 성능을 보였으며, 서포트 세트가 커질수록 그 차이가 뚜렷해졌다.
  - 제안된 방법들은 ADKT-IFT와 같은 최첨단 Meta-learning 방법들과 대등하거나 매우 경쟁력 있는 성능을 기록하였다. (예: $|S|=128$일 때 Quadratic Probe $\Delta$AUCPR 0.310 vs ADKT-IFT 0.318).

### 2. Domain Shift Analysis

- **Prior Shift (DTI tasks)**: 클래스 분포가 매우 불균형한 KIBA, DAVIS, BindingDB_Kd 데이터셋에서 평가하였다.
  - 결과적으로 클래스 불균형이 심해질수록 Meta-learning 방법들의 성능이 하락한 반면, Linear/Quadratic Probe는 훨씬 강건한 성능을 유지하였다.
- **Library Screening**: 히트(Active) 비율이 매우 낮은 ($\approx 1\%$) 극한의 불균형 상황과 타겟 유형이 변경된 상황을 가정하였다.
  - 서포트 세트 크기가 작을 때($|S|=16, 32$) Quadratic Probe가 모든 베이스라인을 압도하는 성능을 보였다.
  - 다만, 서포트 세트가 매우 커지면 사전 훈련의 영향을 받지 않는 Similarity Search(Tanimoto similarity)가 가장 우수한 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 분자 Few-Shot Learning에서 복잡한 Meta-learning이 반드시 최선은 아니며, 적절하게 설계된 Fine-tuning 베이스라인이 더 효율적일 수 있음을 보여주었다.

**강점 및 통찰**:

- **단순성 및 범용성**: 제안된 방법은 표준적인 Multitask 사전 훈련 모델만 있다면 즉시 적용 가능하며, 모델 내부 가중치를 수정하지 않고 예측값만 사용하는 Black-box 설정에서도 Linear/Quadratic probing을 통해 적응할 수 있다.
- **강건성**: Meta-learning 모델들은 훈련 시와 테스트 시의 에피소드 구조가 유사해야 한다는 제약이 있지만, 제안된 Probe 방식은 데이터의 통계적 특성(공분산)을 직접 이용하므로 Prior shift와 같은 도메인 변화에 훨씬 유연하게 대응한다.

**한계 및 논의**:

- **타겟 유형의 영향**: Appendix A.5.1의 결과에서 보듯, 타겟 유형이 완전히 바뀌는 경우(Target type shift)에는 여전히 성능 향상이 어려우며, 이는 사전 훈련된 백본 모델이 학습한 화학적 공간의 한계임을 시사한다.
- **Similarity Search와의 관계**: 데이터가 충분해질수록 단순한 구조적 유사도 검색(Similarity Search)이 더 강력해지는 경향이 있다. 이는 딥러닝 기반 임베딩이 특정 도메인 shift 상황에서는 오히려 노이즈로 작용할 수 있음을 의미한다.

## 📌 TL;DR

본 연구는 복잡한 Meta-learning 대신, **Mahalanobis 거리 기반의 Quadratic Probe**와 이를 안정적으로 최적화하는 **Block-coordinate descent** 알고리즘을 통해 강력한 분자 Few-Shot Learning 베이스라인을 제안하였다. 제안 방법은 기존 Meta-learning 모델들과 대등한 성능을 보이면서도 구현이 훨씬 간단하며, 특히 클래스 불균형이 심한 실제 신약 개발 환경(Domain shift)에서 훨씬 더 강건한 성능을 나타낸다. 이는 향후 분자 기초 모델(Foundational Models)의 Few-shot 적응 능력을 평가하는 데 있어 매우 중요한 기준점이 될 것이다.
