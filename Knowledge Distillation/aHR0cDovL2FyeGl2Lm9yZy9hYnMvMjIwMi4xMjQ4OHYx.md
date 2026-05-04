# Learn From the Past: Experience Ensemble Knowledge Distillation

Chaofei Wang, Shaowei Zhang, Shiji Song, Gao Huang (2022)

## 🧩 Problem to Solve

전통적인 지식 증류(Knowledge Distillation, KD)는 사전 학습된 교사 네트워크(Teacher Network)가 가진 'Dark Knowledge'를 학생 네트워크(Student Network)로 전달하는 데 집중한다. 하지만 이 방식은 교사 네트워크가 최종 모델이 되기까지의 학습 과정에서 축적한 지식, 즉 '교사의 경험(Teacher's Experience)'을 완전히 무시한다는 한계가 있다.

실제 교육 현장에서 학습 결과물보다 학습 과정에서의 경험이 더 중요할 수 있듯이, 딥러닝에서도 교사의 최종 상태뿐만 아니라 학습 경로상의 다양한 상태를 활용하는 것이 더 효과적일 수 있다. 따라서 본 논문은 교사의 학습 과정에서 발생하는 중간 모델들의 지식을 통합하여 학생 네트워크에 전달함으로써, 기존 KD의 한계를 극복하고 학생 모델의 성능을 극대화하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 교사 모델의 학습 과정에서 생성되는 여러 중간 모델들을 앙상블(Ensemble)하여 '가상 교사(Virtual Teacher)'를 구축하고, 이를 통해 학생 모델을 학습시키는 Experience Ensemble Knowledge Distillation (EEKD) 방법론을 제안하는 것이다.

주요 기여 사항은 다음과 같다.
- 교사의 학습 과정에서 적절한 수의 중간 모델을 균일하게 저장하고, 이를 앙상블 기법으로 통합하여 지식을 전달하는 EEKD 프레임워크를 제안하였다.
- 각 중간 모델의 중요도를 동적으로 결정하기 위해 Self-attention 모듈을 도입하여 적응적 가중치(Adaptive Weights)를 할당하는 메커니즘을 설계하였다.
- 앙상블 교사의 성능이 반드시 학생 모델의 성능 향상으로 이어지지 않는다는 역설적인 결론을 도출하여, 기존 앙상블 증류 방식에 대한 새로운 시각을 제시하였다.

## 📎 Related Works

### Knowledge Distillation (KD)
전통적인 KD는 교사와 학생 모델 간의 Soft targets에 대한 KL Divergence를 최소화하는 방식에서 시작되었다. 이후 Fitnets의 중간 레이어 힌트(Hints), AT의 Attention Map, CCKD의 샘플 간 관계 등 다양한 형태의 지식 전달 방식이 제안되었다. 그러나 이러한 연구들은 모두 학습이 완료된 '사전 학습된 교사'에만 의존하며, 학습 과정 중에 형성되는 경험적 지식을 활용하지 않는다는 공통점이 있다.

### Ensemble Learning
여러 모델을 결합하여 예측 정확도를 높이는 앙상블 학습은 널리 사용되지만, 추론 시 계산 비용이 높다는 단점이 있다. 이를 해결하기 위해 여러 교사의 지식을 하나의 학생에게 증류하는 앙상블 증류(Ensemble Distillation) 연구가 진행되었다. 특히 Snapshot Ensemble은 단일 네트워크를 학습시키며 여러 지역 최솟값(Local Minima)에 도달했을 때의 모델들을 저장하여 비용을 줄이는 방식을 제안하였으나, 이를 KD 시나리오에 적용하여 검증한 연구는 부족했다.

### Self-attention
NLP의 Transformer에서 시작된 Self-attention 메커니즘은 이미지 인식 분야(ViT 등)로 확장되었다. 본 논문은 이 메커니즘을 응용하여, 입력 데이터와 학생 모델의 상태에 따라 어떤 중간 교사 모델의 지식이 더 유용한지를 동적으로 결정하는 가중치 학습 모듈로 사용한다.

## 🛠️ Methodology

### 전체 파이프라인
EEKD의 전체 구조는 교사 모델을 학습시키는 단계와, 저장된 중간 모델들을 이용하여 학생 모델을 학습시키는 단계로 나뉜다. 교사 모델 학습 시 전체 과정을 $M$개의 단계로 나누어 균일하게 중간 모델 $\{\theta_{t1}, \theta_{t2}, \dots, \theta_{tM}\}$을 저장한다. 이후 이 모델들을 앙상블하여 하나의 강력한 가상 교사를 구성하고 학생 모델에게 지식을 전수한다.

### 손실 함수 및 학습 절차
전통적인 KD의 손실 함수는 다음과 같다.
$$L^s(B, \theta_s) = -\frac{1}{|B|} \sum_{(x_n, y_n) \in B} \{ \alpha y_n^T \cdot \log f(x_n; \theta_s) + (1-\alpha) KL [ f_\tau(x_n; \theta_t) || f_\tau(x_n; \theta_s) ] \}$$
여기서 $\theta_t$는 단일 교사의 파라미터이다. EEKD에서는 이를 확장하여 교사 부분에 $M$개 모델의 가중 합(weighted sum)을 적용한다.
$$L^s(B, \theta_s) = -\frac{1}{|B|} \sum_{(x_n, y_n) \in B} \{ \alpha y_n^T \cdot \log f(x_n; \theta_s) + (1-\alpha) KL [ (\sum_{i=1}^M w_i f_\tau(x_n; \theta_{ti})) || f_\tau(x_n; \theta_s) ] \}$$
여기서 $w_i$는 각 중간 모델의 가중치이며, $\sum w_i = 1$을 만족해야 한다.

### Attention-based Weighting
단순 평균이나 고정 가중치 대신, 본 논문은 학생 모델과 교사 모델의 특징 벡터를 이용해 가중치를 동적으로 계산한다. 
1. 각 중간 교사 모델의 마지막 컨볼루션 레이어 출력 $u_i$와 학생 모델의 출력 $v$를 선형 변환을 통해 투영한다.
   $$E_s(v) = W_s^T \cdot v, \quad E_t(u_i) = W_t^T \cdot u_i$$
2. 두 벡터 간의 내적을 기반으로 한 Gaussian distance를 계산하고 Softmax를 통해 정규화하여 가중치 $w_i$를 산출한다.
   $$w_i = \frac{e^{E_s(v)^T \cdot E_t(u_i)}}{\sum_{j=1}^M e^{E_s(v)^T \cdot E_t(u_j)}}$$
이 방식은 학생 모델의 학습 단계나 입력 데이터의 특성에 따라 적절한 교사의 지식을 선택적으로 수용할 수 있게 한다.

## 📊 Results

### 실험 설정
- **데이터셋**: CIFAR-100, ImageNet-1K
- **교사-학생 조합**: (ResNet-110 $\rightarrow$ ResNet-20), (WRN-40-2 $\rightarrow$ WRN-40-1), (ResNet-50 $\rightarrow$ MobileNetV2) 등
- **지표**: Top-1 Accuracy (%)
- **하이퍼파라미터**: $\alpha = 0.5, M=5, \tau=5$, Attention-based weight strategy 적용.

### 주요 결과
1. **SOTA 성능 달성**: CIFAR-100 실험에서 EEKD는 KD, FitNet, AT, CRD 등 기존 최신 방법론들을 큰 폭으로 상회하였다. 특히 일부 설정에서는 학생 모델의 성능이 원래 교사 모델의 성능을 뛰어넘는 결과가 나타났다.
2. **ImageNet-1K 검증**: ResNet-34/ResNet-18 조합에서 교사와 학생의 성능 격차를 $3.56\%$에서 $1.64\%$로 줄였으며(상대적 개선 $54\%$), ResNet-50/MobileNetV2 조합에서도 격차를 $11.3\%$에서 $6.7\%$로 줄였다.
3. **표준 앙상블 증류(SED)와의 비교**: $M$개의 독립적인 전체 교사 모델을 학습시켜 증류하는 SED 방식과 비교했을 때, EEKD는 학습 비용을 훨씬 적게 사용하면서도(약 $40\%$ 수준의 비용) 더 높은 정확도를 기록하였다.

## 🧠 Insights & Discussion

### 강점 및 발견
본 논문의 가장 흥미로운 발견은 **"성능이 매우 높은 앙상블 교사가 반드시 성능이 좋은 학생을 만드는 것은 아니다"**라는 점이다. Cyclic cosine learning rate를 사용한 교사 앙상블(Cycle ET)이 일반 교사 앙상블(NoCycle ET)보다 자체 성능은 훨씬 높았지만, 실제 증류 결과는 NoCycle ET를 사용했을 때 더 좋았다. 이는 교사 간의 과도한 다양성(Diversity)이 오히려 인지적 충돌(Cognitive Conflict)을 일으켜 학생의 학습을 방해할 수 있음을 시사한다.

### 분석 및 한계
- **가중치 전략**: Attention 기반 가중치가 고정 가중치(평균, 선형 증가/감소)보다 우수함을 확인하였다. 특히 성능이 높은 후기 모델에 더 높은 가중치를 주는 '선형 증가' 방식이 단순 평균보다 낫지 않았다는 점은, 단순히 성능이 좋은 모델의 지식이 항상 유용한 것은 아님을 의미한다.
- **비용과 성능의 트레이드-오프**: 중간 모델의 수 $M$이 증가할수록 성능은 향상되지만, 학생 학습 시 추론 비용이 선형적으로 증가한다. 따라서 실무적으로는 적절한 $M$ 값의 선택이 필수적이다.

## 📌 TL;DR

본 논문은 교사 모델의 최종 결과뿐만 아니라 학습 과정 중의 중간 상태(Experience)를 앙상블하여 학생에게 전달하는 **Experience Ensemble Knowledge Distillation (EEKD)**를 제안한다. Self-attention 모듈을 통해 각 경험의 가중치를 동적으로 조절함으로써 CIFAR-100 및 ImageNet-1K에서 SOTA 성능을 달성하였다. 특히, 강력한 교사가 반드시 강력한 학생을 만드는 것은 아니라는 통찰을 제시하며, 효율적인 비용으로 고성능의 학생 모델을 구축할 수 있는 새로운 경로를 제시하였다.