# Trustworthy Deep Learning for Medical Image Segmentation

Lucas Fidon (2023)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 딥러닝 방법론이 달성한 높은 평균 정확도에도 불구하고, 실제 임상 현장 적용을 가로막는 **신뢰성(Trustworthiness)**과 **강건성(Robustness)**의 결여 문제를 해결하고자 한다.

구체적으로 해결하려는 문제는 다음과 같다:
1. **데이터의 가변성 및 희소성**: 이미지 획득 프로토콜이나 해부학적 구조의 가변성이 훈련 데이터셋에 충분히 반영되지 않은 경우, 모델이 예상치 못한 치명적인 오류를 범하는 문제가 발생한다.
2. **부분적 주석(Partial Annotation)의 활용 제한**: 의료 영상의 수동 분할은 비용과 시간이 많이 소요되어 모든 관심 영역(ROI)이 완벽하게 분할된 데이터를 얻기 어렵다. 기존의 지도 학습 방식은 모든 영역이 분할된 데이터만을 요구하므로, 일부 영역만 분할된 유용한 데이터를 활용하지 못한다.
3. **숨겨진 계층화(Hidden Stratification)**: 훈련 데이터셋 내에서 특정 하위 집단(underrepresented populations)이 적게 포함된 경우, 모델은 평균 성능은 높지만 해당 소수 집단에 대해서는 매우 낮은 성능을 보이는 '불공정'하고 '위험한' 결과를 낼 수 있다.
4. **해부학적 상식 위배**: 딥러닝 모델이 예측한 결과가 전문가의 해부학적 지식(예: 특정 위치에는 특정 조직이 존재할 수 없음)을 완전히 무시하는 결과가 나올 수 있으며, 이는 임상의의 신뢰를 떨어뜨리는 결정적인 요인이 된다.

본 연구의 최종 목표는 부분적 주석을 활용하고, 최악의 경우(worst-case) 성능을 보장하며, 전문가의 지식을 결합하여 해부학적으로 타당한 결과를 보장하는 **신뢰할 수 있는 딥러닝 분할 프레임워크**를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 딥러닝의 효율성과 전통적인 전문가 지식/통계적 강건성 방법론을 결합하여 '안전 장치'를 마련하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Label-set Loss Functions**: 일부 영역만 분할된 이미지(partially annotated images)를 사용하여 모델을 훈련시킬 수 있는 수학적 프레임워크를 제안하였다. 특히 기존 Dice Loss를 일반화한 `Leaf-Dice` 손실 함수를 통해 데이터 효율성을 극 uma 극대화하였다.
2. **Distributionally Robust Optimization (DRO) 기반 학습**: 평균 손실이 아닌 최악의 경우의 손실을 최소화하는 DRO 방법론을 도입하였다. 이를 위해 계산 효율적인 `Hardness Weighted Sampling` 알고리즘을 제안하여, 훈련 데이터 중 '어려운 샘플'에 더 많은 가중치를 두어 학습함으로써 소수 집단에 대한 강건성을 높였다.
3. **SBA(Spina Bifida Aperta) 태아 뇌 지도(Atlas) 구축**: 정밀한 해부학적 가이드라인을 제공하기 위해, 희귀 질환인 SBA 환아를 위한 시공간적(spatio-temporal) 뇌 지도를 최초로 구축하였다.
4. **Dempster-Shafer 이론 기반의 Trustworthy AI 프레임워크**: 딥러닝 모델(Backbone AI)과 안전한 대체 모델(Fallback algorithm), 그리고 전문가의 지식(Contracts of Trust)을 결합하는 시스템을 제안하였다. AI의 예측이 전문가의 지식(계약)을 위반할 경우, 자동으로 대체 모델의 결과로 전환하는 Fail-safe 메커니즘을 구현하였다.

## 📎 Related Works

논문은 다음과 같은 관련 연구를 검토하고 차별점을 제시한다.

- **도메인 일반화(Domain Generalization)**: 데이터 증강이나 테스트 시간 적응(test-time adaptation)을 통해 강건성을 높이려는 시도가 있었으나, 본 논문은 이를 넘어 '해부학적 계약(contracts of trust)'이라는 명시적인 제약 조건을 도입하여 신뢰성을 확보하고자 한다.
- **태아 뇌 MRI 분할**: 기존의 Atlas 기반 방법은 해부학적으로 타당하지만 정확도가 낮고, 딥러닝 방법은 정확도는 높지만 강건성이 낮다. 본 논문은 이 두 가지의 장점을 Dempster-Shafer 이론으로 결합하여 상호 보완적인 시스템을 구축하였다.
- **부분 지도 학습(Partial Supervision)**: 기존 연구들이 임의의 marginalization 방법을 사용했다면, 본 논문은 'label-set inclusion'이라는 공리를 정의하고 이를 만족하는 손실 함수의 수학적 체계를 정립하여 이론적 근거를 마련하였다.

## 🛠️ Methodology

본 논문은 크게 세 가지 기술적 방법론을 제안한다.

### 1. 부분 감독 학습을 위한 Label-set Loss Functions
모든 픽셀이 단일 레이블(leaf-label)을 갖는 대신, 레이블의 집합(label-set)을 가질 수 있도록 정의한다. 예를 들어, 어떤 픽셀이 '피질 회백질'인지 '심부 회백질'인지 불분명할 때 이를 $\{CGM, DGM\}$이라는 집합으로 표현한다.

- **Leaf-Dice Loss**: 
  $$L_{Leaf-Dice}(p,g) = 1 - \frac{1}{|L|} \sum_{c \in L} \frac{2 \sum_{i} \mathbb{1}(g_i=\{c\})p_{i,c}}{\sum_{i} \mathbb{1}(g_i=\{c})^\alpha + \sum_{i} p_{i,c}^\alpha + \epsilon}$$
  이 식은 주석이 없는 영역을 단순히 무시하는 것이 아니라, 해당 영역의 확률 합이 1이 되어야 한다는 제약 조건을 통해 간접적으로 학습에 참여시킨다.

### 2. Distributionally Robust Optimization (DRO) 및 Hardness Weighted Sampling
평균 위험 최소화(ERM) 대신, 데이터 분포의 불확실성을 고려하여 최악의 경우의 성능을 최적화한다.

- **최적화 목표**:
  $$\min_{\theta} \max_{q \in \Delta_n} \left( \mathbb{E}_{(x,y) \sim q} [L(h(x;\theta), y)] - \frac{1}{\beta} D_\phi(q \| p_{train}) \right)$$
  여기서 $D_\phi$는 $\phi$-divergence이며, $\beta$는 강건성 조절 파라미터이다.
- **학습 절차 (Hardness Weighted Sampling)**:
  매 반복마다 모든 샘플의 손실 값을 계산하는 대신, 'stale'한(이전 단계의) 손실 값을 저장해두고 다음과 같은 확률로 샘플을 추출한다.
  $$p_t = \text{softmax}(\beta L)$$
  즉, 손실 값이 큰(어려운) 샘플이 더 자주 선택되어 학습되도록 함으로써, 모델이 소수 집단의 특징을 더 잘 학습하게 만든다.

### 3. Dempster-Shafer 이론 기반 Trustworthy AI
AI 모델의 예측 결과가 전문가의 지식(Anatomical/Intensity priors)과 충돌할 때 이를 교정하는 시스템이다.

- **전체 파이프라인**:
  $$p_{TWAI} = \left( (1-\epsilon)p_{AI} + \epsilon p_{fallback} \right) \oplus m_{fail-safe}$$
  여기서 $\oplus$는 Dempster의 결합 법칙(Dempster's rule of combination)이다.
- **작동 원리**:
  - **Backbone AI**: 고성능 딥러닝 모델 (예: nnU-Net).
  - **Fallback**: Atlas 기반의 보수적이고 안정적인 모델.
  - **Fail-safe ($m_{fail-safe}$)**: "흰색질은 이 영역 밖에 존재할 수 없다"와 같은 해부학적 계약.
  - 만약 $p_{AI}$가 전문가의 지식 $m_{fail-safe}$와 완전히 모순된다면, 결합 법칙에 의해 결과값은 $\epsilon$과 관계없이 $p_{fallback}$의 결과로 완전히 전환된다.

## 📊 Results

### 실험 설정
- **데이터셋**: 6개국 13개 병원에서 수집한 540개의 태아 뇌 3D MRI (현존 최대 규모).
- **평가 지표**: Dice Score (DSC), Hausdorff Distance (HD95), 전문의 신뢰도 점수 (0-5점).
- **비교 대상**: nnU-Net (ERM), 단순 Atlas 기반 방법.

### 주요 결과
1. **부분 감독 학습**: `Leaf-Dice`를 사용했을 때, 일부 영역만 주석된 데이터를 추가로 사용함으로써 전체적인 분할 정확도가 유의미하게 향상되었다. 특히 데이터가 부족한 조직(예: 뇌줄기, 심부 회백질)에서 효과가 컸다.
2. **DRO의 효과**: DRO를 적용한 모델은 평균 성능은 ERM과 비슷하지만, **최악의 경우의 성능(5th percentile Dice)**이 크게 개선되었다. 특히 SBA와 같은 이상 해부학 구조를 가진 케이스에서 뇌소뇌(Cerebellum)를 완전히 놓치는 오류가 크게 감소하였다.
3. **Trustworthy AI의 강건성**: 
   - **정량적 결과**: $\text{TW-AI}$는 $p_{AI}$보다 HD95(거리 오차)가 훨씬 낮았으며, $p_{fallback}$보다 DSC(정확도)가 높았다.
   - **정성적 결과**: 전문가 8인이 평가한 신뢰도 점수에서 $\text{TW-AI}$가 가장 높은 점수를 받았으며, 특히 훈련 데이터에 없었던 새로운 병원(out-of-scanner)의 데이터에서도 해부학적 오류가 거의 발견되지 않았다.

## 🧠 Insights & Discussion

### 강점 및 의의
- **상호 보완적 결합**: 딥러닝의 '높은 정확도'와 Atlas의 '해부학적 안정성'을 수학적(DS 이론)으로 결합하여, AI의 고질적인 문제인 '말도 안 되는 예측(spectacular failures)'을 효과적으로 차단하였다.
- **이론적 뒷받침**: 단순히 휴리스틱한 방법을 쓴 것이 아니라, Label-set loss의 공리 정의, DRO의 수렴성 증명 등 학술적 깊이를 갖추었다.

### 한계 및 미해결 질문
- **Atlas 의존성**: 본 시스템의 신뢰성은 기반이 되는 Atlas의 품질에 의존한다. 따라서 Atlas가 존재하지 않는 매우 희귀한 질환에 대해서는 여전히 취약할 수 있다.
- **계산 비용**: Multi-atlas 기반의 fallback 모델을 사용하므로, 추론 시 여러 개의 이미지를 등록(registration)해야 하는 계산적 오버헤드가 존재한다.

### 비판적 해석
본 연구는 AI를 완전히 대체하는 것이 아니라, 전문가의 지식을 '필터'로 사용하여 AI의 결과물을 검증하는 구조를 취하고 있다. 이는 의료 분야에서 AI를 도입할 때 가장 중요한 '안전성'과 '설명 가능성'을 실무적인 수준에서 구현한 매우 현실적인 접근법이라고 평가할 수 있다.

## 📌 TL;DR

본 논문은 의료 영상 분할에서 딥러닝 모델이 보이는 **불안정한 강건성과 해부학적 오류**를 해결하기 위해 세 가지 전략을 제안한다. 
1. **부분 주석 활용**을 위한 `Leaf-Dice` 손실 함수 제안.
2. **최악의 성능 개선**을 위한 `Hardness Weighted Sampling` 기반의 DRO 학습법 제안.
3. **전문가 지식 기반의 Fail-safe**를 위해 Dempster-Shafer 이론을 이용해 AI와 Atlas 모델을 결합한 Trustworthy AI 프레임워크 구축.
결과적으로, 이 시스템은 특히 희귀 질환 및 새로운 환경의 데이터에서 **해부학적으로 타당하고 신뢰할 수 있는 분할 결과**를 제공하며, 이는 향후 의료 AI의 임상 적용을 위한 중요한 안전장치가 될 가능성이 높다.