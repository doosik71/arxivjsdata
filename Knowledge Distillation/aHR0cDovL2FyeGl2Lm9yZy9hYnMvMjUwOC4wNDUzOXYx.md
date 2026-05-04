# TopKD: Top-scaled Knowledge Distillation

Qi Wang, Jinjia Zhou (2025)

## 🧩 Problem to Solve

본 논문은 지식 증류(Knowledge Distillation, KD) 과정에서 교사 모델(Teacher model)의 로짓(Logit) 분포에 포함된 핵심 정보가 충분히 활용되지 못하고 있다는 문제에서 출발한다. 최근의 KD 연구들은 주로 중간 레이어의 특징 맵(Feature map)을 전달하는 Feature-level KD에 집중해 왔으며, 이는 풍부한 시맨틱 표현을 전달할 수 있지만 높은 계산 비용과 복잡한 훈련 절차, 그리고 교사와 학생 모델 간의 아키텍처 의존성이라는 한계를 가진다.

반면, Logit-based KD는 가볍고 유연하여 다양한 모델에 적용 가능하지만, 일반적으로 Feature-based KD보다 성능이 낮다. 저자들은 그 원인이 최종 출력 레이어의 정보 병목 현상과 더불어, 기존에 널리 사용된 Kullback-Leibler divergence (KL-Div) 손실 함수가 교사와 학생의 로짓 분포를 너무 엄격하게 일치시키려 하기 때문이라고 분석한다. 이러한 경직된 정렬은 지식 전송의 풍부함을 제한하고 학생 모델의 일반화 성능을 저해할 수 있다. 따라서 본 논문의 목표는 교사 모델의 로짓 중 특히 정보량이 많은 Top-K 지식을 효과적으로 추출하고 활용하여, Feature-based KD에 필적하거나 이를 능가하는 효율적인 Logit-based KD 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 교사 모델의 로짓 분포에서 상위 $K$개의 값(Top-K knowledge)이 클래스 간의 시맨틱 관계를 포함하는 매우 중요한 감독 신호라는 점에 주목한 것이다. 이를 위해 제안된 **TopKD** 프레임워크의 주요 기여 사항은 다음과 같다.

1.  **Top-K Scaling Module (TSM) 제안**: 교사 모델의 로짓 중 가장 정보량이 많은 Top-K 성분을 적응적으로 증폭시켜, 학습 과정에서 이들의 영향력을 강조한다. 특히 교사 모델의 예측이 틀렸을 경우 정답(Ground-truth) 로짓에 더 큰 스케일링을 적용하여 바이어스를 교정한다.
2.  **Top-K Decoupled Loss (TDL) 제안**: 단순한 값의 일치가 아닌, 임베딩 공간에서의 방향성 일치를 위해 코사인 유사도(Cosine Similarity)를 활용한다. 이때 로짓을 Positive Top-K, Negative Top-K, Non-Top-K의 세 부분으로 분리하여 각각 다른 가중치를 부여함으로써 구조적 일관성을 정밀하게 학습한다.
3.  **KL-Div 대체 및 Contrastive Learning 도입**: 엄격한 분포 일치 대신, 샘플 간의 상대적 관계를 학습하는 Contrastive Loss를 도입하여 로짓의 구조적 정보를 더 효과적으로 캡처한다.
4.  **범용적 플러그 앤 플레이(Plug-and-Play) 설계**: TopKD는 특정 아키텍처에 종속되지 않으며, 기존의 다른 KD 방법론에 추가 모듈 없이 통합되어 성능을 향상시킬 수 있는 구조를 가진다.

## 📎 Related Works

### 기존 연구 및 한계
- **Logit-based KD**: Hinton 등이 제안한 전통적인 KD는 KL-Div를 사용하여 교사와 학생의 출력 확률 분포를 맞춘다. 이는 계산 효율적이지만 교사 모델 내부의 풍부한 표현 구조를 충분히 전달하지 못하는 한계가 있다.
- **Feature-based KD**: 중간 레이어의 특징을 전송하여 더 풍부한 감독을 제공하지만, 교사와 학생의 아키텍처가 유사해야 하거나 차원 맞춤을 위한 추가 모듈이 필요하며 계산 비용이 크다.
- **Contrastive Learning**: SimCLR, MoCo 등에서 사용되는 대조 학습은 긍정 쌍(Positive pair)은 가깝게, 부정 쌍(Negative pair)은 멀게 배치함으로써 강건한 표현 학습을 가능하게 한다. 일부 KD 연구에서 이를 도입했으나, 로짓의 세부적인 Top-K 구조를 직접적으로 다루지는 않았다.

### TopKD의 차별점
TopKD는 단순히 로짓의 값을 맞추는 것이 아니라, Top-K 지식이라는 시맨틱 구조에 집중한다. 특히 KL-Div의 경직성을 탈피하여 Contrastive Loss와 Decoupled Cosine Similarity를 결합함으로써, 로짓 수준에서도 Feature-based KD 수준의 풍부한 구조적 정보를 전송할 수 있음을 보여준다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 구조
TopKD는 교사 모델의 로짓을 입력받아 $\text{TSM} \rightarrow \text{TDL}$ 순으로 처리하며, 최종적으로 Contrastive Loss와 TDL을 결합한 총 손실 함수를 통해 학생 모델을 학습시킨다.

### 2. Contrastive Loss
인스턴스 수준의 정렬을 위해 Contrastive Loss를 도입한다. 학생 모델의 로짓 $z_s$와 교사 모델의 로짓 $z_t$ 사이의 유사도 행렬을 계산하여, 동일 샘플의 쌍(대각 성분)은 최대화하고 서로 다른 샘플의 쌍(비대각 성분)은 최소화한다.

$$L_{Contrastive} = \frac{1}{2} \left( L_{CE}((z_s * z_t^\top)/\tau, y) + L_{CE}((z_t * z_s^\top)/\tau, y) \right)$$

여기서 $L_{CE}$는 Cross Entropy 손실이며, $\tau$는 유사도 점수의 날카로움을 조절하는 온도 매개변수이다. $y$는 배치 내 인덱스를 나타낸다.

### 3. Top-K Scaling Module (TSM)
교사 모델의 로짓 중 Top-K에 해당하는 인덱스 $I_{top}$와 정답 레이블 $y_g$에 대해 다음과 같이 값을 재스케일링한다.

$$z'_i = \begin{cases} z_i \cdot w_i + \Delta, & \text{if } i \in I_{top} \cup \{y_g\} \\ z_i, & \text{otherwise} \end{cases}$$

여기서 $w_i$는 순위에 따른 스케일링 계수이며, $\Delta$는 Top-K와 Non-Top-K 로짓 간의 평균 차이에 비례하는 바이어스 항이다. 이를 통해 시맨틱하게 관련성이 높은 카테고리의 중요도를 높이고 노이즈를 억제한다.

### 4. Top-K Decoupled Loss (TDL)
단순한 전체 코사인 유사도는 정보량이 적은 로짓에 의해 희석될 수 있다. 이를 해결하기 위해 교사 로짓의 부호와 크기에 따라 세 그룹으로 분리하여 유사도를 계산한다.
- **Positive Top-K**: 가장 큰 양수 값들을 가진 차원.
- **Negative Top-K**: 가장 작은 음수 값들을 가진 차원.
- **Non-Top-K**: 나머지 차원.

$$L_{TDL} = 1 - \frac{1}{B} \sum_{i=1}^{B} \left[ \alpha \cdot \cos(z_{s,i}^{Pos}, z_{t,i}^{\prime Pos}) + \beta \cdot \cos(z_{s,i}^{Neg}, z_{t,i}^{\prime Neg}) + \cos(z_{s,i}^{Non}, z_{t,i}^{\prime Non}) \right]$$

여기서 $\alpha, \beta$는 각 그룹의 중요도를 조절하는 가중치이다.

### 5. 최종 학습 목표
최종 손실 함수는 다음과 같이 정의되며, 인스턴스 간 정렬(Contrastive)과 인스턴스 내부 구조 정렬(TDL)을 동시에 최적화한다.

$$L_{TopKD} = L_{Contrastive} + L_{TDL}$$

## 📊 Results

### 실험 설정
- **데이터셋**: CIFAR-100, ImageNet, STL-10, Tiny-ImageNet.
- **비교 대상**:
    - Feature-based: CRD, ReviewKD, SimKD, CAT-KD, FCFD.
    - Logit-based: KD, DKD, DOT, LSKD, WTTM.
- **모델**: ResNets, WRNs, VGGs, MobileNets, ShuffleNets 및 ViT 계열 (DeiT-Ti, T2T-ViT 등).

### 주요 결과
- **성능 우위**: CIFAR-100 및 ImageNet 데이터셋에서 TopKD는 대부분의 Logit-based 및 Feature-based 방법론보다 높은 Top-1 정확도를 기록하였다. 특히 ImageNet(ResNet-50 $\rightarrow$ MobileNetV1) 설정에서 $73.51\%$의 정확도를 달성하며 SOTA 성능을 보였다.
- **Top-K 값의 영향**: $K=3, 5, 10$일 때 성능이 가장 좋았으며, $K$가 너무 커지면(예: $K=50$) 오히려 성능이 급격히 하락($60.70\%$)하는 것을 확인하였다. 이는 Top-K 지식의 희소성과 중요성을 입증한다.
- **플러그 앤 플레이 능력**: TSM과 TDL 모듈을 기존 KD 방법론에 통합했을 때, 모든 경우에서 일관된 성능 향상이 관찰되었다.
- **전이 가능성(Transferability)**: CIFAR-100에서 학습된 모델을 STL-10 및 Tiny-ImageNet으로 전이했을 때, TopKD가 가장 우수한 표현력을 보였다.
- **교사 모델 용량 문제 완화**: 일반적으로 너무 강력한 교사 모델은 학생 모델과의 용량 차이(Capacity Gap)로 인해 효율적인 전송이 어려우나, TopKD는 교사 모델의 성능이 높아짐에 따라 학생 모델의 성능도 일관되게 향상되는 경향을 보였다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 논문은 Logit-based KD가 성능이 낮은 이유가 단순한 값의 일치(Value matching)에 매몰되었기 때문임을 밝혀냈으며, 이를 임베딩 공간에서의 구조적 방향성 일치(Directional consistency) 문제로 전환하여 해결하였다. 특히 TSM을 통해 교사 모델의 잘못된 예측(Misclassification) 속에 숨겨진 시맨틱 관계(예: 물개-수달)를 포착하여 학생에게 전달함으로써, 단순한 정답 학습 이상의 일반화 능력을 부여한 점이 돋보인다.

### 한계 및 논의사항
- **하이퍼파라미터 민감도**: $K$ 값과 $\alpha, \beta$ 가중치에 따라 성능 차이가 발생하므로, 다양한 데이터셋과 아키텍처에 적용하기 위한 최적의 하이퍼파라미터 탐색 방법론이 추가로 필요할 수 있다.
- **계산 복잡도**: 로짓 기반이므로 Feature-based보다는 훨씬 가볍지만, 배치 크기가 커질 때 Contrastive Loss의 유사도 행렬 계산 비용이 증가할 가능성이 있다.
- **가정**: 본 논문은 Top-K 로짓이 항상 의미 있는 시맨틱 정보를 담고 있다고 가정한다. 하지만 교사 모델이 완전히 잘못된 예측을 하는 경우, 이 Top-K 정보가 오히려 노이즈로 작용할 가능성에 대한 심층적인 분석이 부족하다.

## 📌 TL;DR

TopKD는 교사 모델의 로짓 분포 중 가장 정보량이 많은 **Top-K 지식**을 선택적으로 증폭(TSM)하고 구조적으로 정렬(TDL)하는 단순하고 효율적인 KD 프레임워크이다. 기존의 KL-Div 대신 Contrastive Loss와 분리된 코사인 유사도를 사용하여 로짓의 시맨틱 구조를 보존하며, 이는 Feature-based KD의 높은 성능과 Logit-based KD의 효율성을 동시에 잡은 결과이다. 특히 다양한 모델 아키텍처(CNN, ViT)와 작업(분류, 탐지)에 범용적으로 적용 가능하여, 향후 경량 모델 학습을 위한 핵심적인 로짓 처리 기법으로 활용될 가능성이 높다.