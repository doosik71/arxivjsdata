# What’s Left After Distillation? How Knowledge Transfer Impacts Fairness and Bias

Aida Mohammadshahi, Yani Ioannou (2025)

## 🧩 Problem to Solve

본 논문은 딥러닝 모델 압축 기법으로 널리 사용되는 Knowledge Distillation(지식 증류, 이하 KD)이 모델의 **알고리즘 편향(Algorithmic Bias)과 공정성(Fairness)**에 미치는 영향력을 분석한다.

일반적으로 KD는 거대한 Teacher 모델의 지식을 작은 Student 모델로 전이하여, 모델 크기를 줄이면서도 전체적인 일반화 성능(Generalization Performance)을 유지하는 데 집중한다. 그러나 대부분의 기존 연구는 전체 정확도(Overall Accuracy)의 유지 여부만을 평가할 뿐, KD 과정이 각 클래스별 정확도에 불균일하게 영향을 미치는지, 혹은 특정 인구통계학적 그룹에 대한 편향을 심화하거나 완화하는지에 대해서는 간과해 왔다.

따라서 본 연구의 목표는 KD가 클래스별 정확도(Class-wise Accuracy)에 미치는 통계적 영향력을 조사하고, 특히 KD의 핵심 하이퍼파라미터인 **온도(Temperature, $T$)**가 모델의 그룹 공정성(Group Fairness) 및 개별 공정성(Individual Fairness)에 어떠한 변화를 일으키는지 정량적으로 분석하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여 및 직관은 다음과 같다.

1.  **KD의 불균일한 영향력 증명**: KD가 모든 클래스의 성능을 균일하게 변화시키는 것이 아니라, 특정 클래스들에 선택적으로 영향을 미친다는 점을 통계적으로 입증하였다.
2.  **온도($T$)와 편향의 상관관계 규명**: KD의 온도 설정이 Student 모델의 편향 방향을 결정함을 밝혔다. 낮은 온도는 Non-distilled Student(학습 데이터로만 학습한 모델)의 편향과 유사하게 만들고, 높은 온도는 Teacher 모델의 편향과 더 유사하게 만든다.
3.  **공정성 개선 가능성 제시**: 적절한 범위 내에서 KD 온도를 높이면 Student 모델의 그룹 공정성 및 개별 공정성이 향상되며, 심지어 특정 조건에서는 Teacher 모델보다 더 높은 공정성을 달성할 수 있음을 보여주었다.
4.  **멀티모달 검증**: 이미지 분류(CV)뿐만 아니라 텍스트 분류(NLP) 데이터셋에서도 동일한 경향성이 나타남을 확인하여 연구 결과의 일반성을 확보하였다.

## 📎 Related Works

### 기존 연구 및 한계
모델 압축 기법 중 Pruning(가지치기)과 Quantization(양자화)이 모델의 알고리즘 편향을 악화시킬 수 있다는 연구(Hooker et al., 2019)는 존재했으나, KD에 대한 논의는 상대적으로 부족했다. 기존의 KD 관련 연구들은 주로 성능 최적화에 치중했으며, 공정성 관점에서의 분석은 매우 제한적이었다. 일부 연구(Chai et al., 2022)에서 손실 함수의 가중치 $\alpha$가 공정성에 영향을 준다는 점을 언급했으나, 온도 파라미터 $T$의 역할을 체계적으로 분석한 연구는 없었다.

### 본 논문의 차별점
본 연구는 단순히 성능 지표를 확인하는 것을 넘어, **Welch's t-test**와 같은 통계적 검증을 통해 클래스별 영향력을 분석하였고, DPD, EOD 및 Lipschitz condition 기반의 개별 공정성 지표를 도입하여 KD의 영향력을 다각도로 분석했다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 학습 절차
본 연구는 Teacher 모델이 생성한 **Soft Targets**를 Student 모델이 학습하도록 하는 전형적인 KD 구조를 따른다. 비교를 위해 다음 세 가지 모델을 설정한다.
- **Teacher**: 사전 학습된 거대 모델.
- **Non-distilled Student (NDS)**: KD 없이 Hard Target(정답 레이블)만으로 학습한 작은 모델.
- **Distilled Student (DS)**: Teacher로부터 지식을 전이받아 학습한 작은 모델.

### 2. 핵심 방정식 및 손실 함수
**온도($T$)가 적용된 Softmax**:
Teacher 모델의 로짓(Logits) $z_i$를 확률 분포로 변환할 때, 온도 파라미터 $T$를 도입하여 분포를 부드럽게(soften) 만든다.
$$p_i = \frac{e^{z_i/T}}{\sum_{j} e^{z_j/T}}$$
$T=1$일 때 일반적인 Softmax와 같으며, $T$가 증가할수록 확률 분포의 엔트로피가 높아져 정답 외의 클래스들이 가진 상대적 정보가 더 많이 전달된다.

**전체 손실 함수**:
Student 모델은 다음과 같이 두 손실 함수의 가중 합을 최소화하도록 학습된다.
$$L_{total} = \alpha \times L_{Distillation} + (1 - \alpha) \times L_{Classification}$$
여기서 $L_{Distillation}$은 Teacher의 soft targets와의 Cross-entropy이며, $L_{Classification}$은 실제 정답(hard targets)과의 Cross-entropy이다.

### 3. 공정성 측정 지표
- **Demographic Parity Difference (DPD)**: 서로 다른 인구통계학적 그룹 $A$ 간의 긍정 예측 확률 차이의 최대-최소값이다.
  $$DPD = \max_{a \in A} P(\hat{Y}=1|A=a) - \min_{a \in A} P(\hat{Y}=1|A=a)$$
- **Equalized Odds Difference (EOD)**: True Positive Rate(TPR) 차이와 False Positive Rate(FPR) 차이 중 더 큰 값을 선택한다.
  $$EOD = \max(\text{TPR Difference}, \text{FPR Difference})$$
- **Individual Fairness (개별 공정성)**: 유사한 입력 $x, x'$에 대해 유사한 예측을 내놓는지 측정하는 Lipschitz condition을 사용한다.
  $$I(f) = \mathbb{E}_{(x, x') \sim P} \left[ \frac{|f(x) - f(x')|}{d(x, x')} \right]$$
  여기서 $d(x, x')$는 사전 학습된 특징 공간에서의 코사인 유사도 기반 거리이다.

### 4. 통계적 유의성 검정
KD가 특정 클래스에 편향된 영향을 주었는지 확인하기 위해 **Welch's t-test**를 사용한다. 
- **귀무가설($H_0$)**: 클래스별 정확도의 변화율이 전체 모델 정확도의 변화율과 동일하다 (즉, 영향이 균일하다).
- **대립가설($H_1$)**: 특정 클래스의 정확도 변화가 전체 평균 변화와 통계적으로 다르다 (즉, 편향된 영향이 존재한다).

## 📊 Results

### 1. 클래스별 편향 분석 (Class-wise Bias)
- **데이터셋**: CIFAR-10/100, SVHN, Tiny ImageNet, ImageNet.
- **결과**: 
    - KD는 모든 클래스에 균일하게 작용하지 않는다. CIFAR-100과 ImageNet의 경우, 최대 41%의 클래스가 통계적으로 유의미하게 영향을 받았다.
    - **온도($T$)의 영향**: $T$가 증가할수록 NDS 모델 대비 유의미하게 변화하는 클래스의 수(#SC)는 증가하고, Teacher 모델 대비 유의미하게 차이 나는 클래스의 수(#TC)는 감소한다. 즉, 온도가 높을수록 Student는 Teacher의 편향을 더 강하게 복제한다.

### 2. 그룹 및 개별 공정성 분석
- **데이터셋**: CelebA(이미지), Trifeature(합성 이미지), HateXplain(텍스트).
- **결과**:
    - **온도와 공정성**: $T$를 $2$에서 $10$까지 높일 때, DPD와 EOD 수치가 전반적으로 감소(공정성 향상)하는 경향을 보였다.
    - **Teacher 초과 달성**: 일부 설정(예: HateXplain의 성별 속성)에서는 Distilled Student가 Teacher 모델보다 더 낮은 DPD/EOD를 기록하며 더 공정한 예측을 수행하였다.
    - **개별 공정성**: $T$가 증가함에 따라 Lipschitz 기반 개별 공정성 점수 또한 개선되었다.

### 3. 한계 지점 (Extreme Temperatures)
- $T \in \{20, 30, 40\}$과 같은 극단적인 고온 설정에서는 Teacher의 확률 분포가 거의 균등 분포(Uniform Distribution)에 가까워져 유용한 정보가 전달되지 않는다. 이 경우 전체 정확도가 하락하며 공정성 지표 또한 다시 악화되는 양상을 보였다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 논문은 KD가 단순히 모델 크기를 줄이는 중립적인 도구가 아니라, 모델의 의사결정 경계와 편향을 재구성하는 과정임을 입증하였다. 특히 **"온도($T$) 조절을 통해 모델의 공정성을 제어할 수 있다"**는 발견은 매우 실무적인 가치가 있다. 정확도와 공정성 사이에 트레이드-오프(Trade-off)가 존재하므로, 적용 도메인의 민감도에 따라 최적의 $T$를 선택해야 함을 시사한다.

### 한계 및 비판적 해석
- **데이터셋의 특성**: CelebA와 같은 실제 데이터셋은 태생적인 불균형(Long-tail)을 가지고 있어, KD가 이를 완화하는 것인지 아니면 단순히 특정 그룹의 예측 확률을 낮추어 수치적 공정성만 맞춘 것인지에 대한 더 깊은 분석이 필요하다.
- **$\alpha$ 값의 고정**: 본 실험에서는 $\alpha=0.8$을 주로 사용하였으나, $\alpha$와 $T$의 상호작용이 공정성에 미치는 영향에 대해서는 충분히 다루지 않았다.

## 📌 TL;DR

본 연구는 Knowledge Distillation(KD)이 모델의 클래스별 정확도에 불균일한 영향을 미치며, 이는 모델의 편향과 공정성으로 이어진다는 것을 밝혔다. 특히 **KD의 온도($T$)를 높이면(적정 범위 내에서) Student 모델의 그룹 및 개별 공정성이 향상**되며, 때로는 Teacher 모델보다 더 공정한 모델을 만들 수 있음을 보였다. 이는 민감한 도메인에 모델 압축을 적용할 때, 단순 정확도뿐만 아니라 온도 설정에 따른 공정성 변화를 반드시 고려해야 함을 시사한다.