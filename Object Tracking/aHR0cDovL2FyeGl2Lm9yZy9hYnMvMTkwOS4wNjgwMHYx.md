# GradNet: Gradient-Guided Network for Visual Object Tracking

Peixia Li, Boyu Chen, Wanli Ouyang, Dong Wang, Xiaoyun Yang, Huchuan Lu (2019)

## 🧩 Problem to Solve

본 논문은 Visual Object Tracking 분야에서 Siamese 네트워크 기반 트래커들이 가지는 한계점을 해결하고자 한다. Siamese 네트워크(예: SiameseFC)는 오프라인 학습을 통해 템플릿 매칭 능력을 기르며, 테스트 단계에서는 첫 프레임의 타겟 특징을 템플릿으로 고정하여 사용한다. 이러한 방식은 실시간 속도를 보장하지만, 다음과 같은 치명적인 문제가 발생한다.

1.  **템플릿 고정으로 인한 적응력 부족**: 타겟의 외형 변화(temporal variations)나 배경의 복잡함(background clutter)에 대응하지 못해 추적 실패(tracking drift)의 위험이 크다.
2.  **온라인 업데이트의 트레이드-오프**: 기존의 온라인 업데이트 방식 중 Gradient-descent 기반 방식은 높은 정확도를 제공하지만, 수백 번의 반복 계산(iterations)이 필요하여 실시간성(real-time)을 충족하지 못한다. 반면, 템플릿 조합(template combination) 방식은 속도는 빠르나 배경의 판별적 정보(discriminative information)를 무시하는 경향이 있다.

따라서 본 연구의 목표는 Gradient-descent 방식의 판별적 정보 활용 능력과 Siamese 네트워크의 실시간 속도를 동시에 확보할 수 있는 새로운 템플릿 업데이트 메커니즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Gradient의 판별적 정보를 활용하여 템플릿을 업데이트하는 CNN 기반의 GradNet을 설계**하는 것이다. 

가장 중심적인 직관은 Gradient가 타겟의 변화와 배경 노이즈를 반영하는 풍부한 정보를 담고 있다는 점이다. 기존의 수동 설계된 최적화 알고리즘(SGD, ADAM 등) 대신, 비선형 함수인 CNN을 통해 Gradient 기반의 최적화 과정을 모사함으로써, 단 한 번의 역전파(backward propagation)와 두 번의 순전파(feed-forward)만으로 효율적인 템플릿 업데이트를 수행하고자 한다.

또한, 네트워크가 단순히 외형 정보에만 의존하거나 과적합(overfitting)되는 것을 방지하기 위해 **Template Generalization**이라는 특수한 학습 방법을 제안하였다.

## 📎 Related Works

### 1. Siamese Network based Tracking
SiameseFC, SINT, GOTURN 등이 대표적이다. 이들은 오프라인 학습 후 고정된 템플릿을 사용하므로 속도가 매우 빠르지만, 온라인 적응 능력이 없어 외형 변화에 취약하다.

### 2. Model Updating in Tracking
- **Template Combination**: 이전 프레임의 특징들을 결합하는 방식으로 속도는 빠르나 배경의 판별적 정보를 무시한다.
- **Gradient-descent based**: 역전파 Gradient를 통해 모델을 업데이트하여 정확도가 높지만, 수렴까지 많은 반복 계산이 필요해 실시간성이 떨어진다.
- **Correlation based**: 푸리에 도메인에서 빠르게 계산하지만, 딥러닝 네트워크의 템플릿 업데이트 과정을 완전히 모사하기 어렵다.

### 3. Gradient Exploiting & Meta Learning
본 연구는 최적화 기반의 Meta Learning과 유사한 구조를 가지나, 다음의 차별점을 갖는다.
- 전체 네트워크가 아닌 **템플릿 업데이트**에만 집중하여 추적 작업에 최적화하였다.
- 여러 번의 반복 대신 **단 한 번의 iteration**으로 업데이트를 완료한다.
- 파라미터 학습 과정에서 **2차 Gradient(second-order gradient)**를 활용한다.

## 🛠️ Methodology

### 전체 파이프라인
GradNet은 두 개의 브랜치로 구성된다. 하나는 검색 영역(search region $X$)의 특징을 추출하는 브랜치이고, 다른 하나는 타겟 정보와 Gradient를 이용해 템플릿을 생성하는 업데이트 브랜치(update branch)이다.

### 템플릿 생성 과정 (Template Generation)
템플릿 생성은 다음의 세 단계로 이루어진다.

**1. Initial Embedding (초기 임베딩)**
얕은 타겟 특징 $f_2(Z)$를 서브넷 $U_1$에 통과시켜 초기 템플릿 $\beta$를 생성한다.
$$\beta = U_1(f_2(Z), \alpha_1)$$
여기서 $\alpha_1$은 $U_1$의 파라미터이다. 이 $\beta$를 이용하여 검색 영역 $f_x(X)$와 교차 상관 연산(cross correlation)을 수행해 초기 스코어 맵 $S$를 얻는다.
$$S = \beta * f_x(X)$$

**2. Gradient Calculation (Gradient 계산)**
초기 스코어 맵 $S$와 정답 라벨 $Y$ 사이의 Logistic Loss $L = l(S, Y)$를 계산한다. 이 손실 함수를 통해 얕은 타겟 특징 $f_2(Z)$에 대한 Gradient를 구하고, 이를 서브넷 $U_2$를 통해 비선형 변환하여 타겟 특징을 업데이트한다.
$$h_2(Z) = f_2(Z) + U_2\left(\frac{\partial L}{\partial f_2(Z)}, \alpha_2\right)$$
이 과정에서 $\frac{\partial L}{\partial f_2(Z)}$가 $U_1$과 연관되어 있으므로, $U_1$의 파라미터 학습 시 2차 Gradient 정보가 반영된다.

**3. Template Update (템플릿 업데이트)**
업데이트된 특징 $h_2(Z)$를 다시 $U_1$에 통과시켜 최적의 템플릿 $\beta^*$와 최종 스코어 맵 $S^*$를 생성한다.
$$\beta^* = U_1(h_2(Z), \alpha_1)$$
$$S^* = \beta^* * f_x(X)$$
최종 목표는 $\sum l(S^*, Y)$를 최소화하도록 업데이트 브랜치의 파라미터 $\alpha$를 학습시키는 것이다.

### Template Generalization (템플릿 일반화 학습)
단순한 학습 방식은 네트워크가 Gradient보다는 단순 외형(appearance)에 의존하게 만들며, 이는 과적합으로 이어진다. 이를 해결하기 위해 저자들은 다음과 같은 일반화 전략을 제안한다.

- **학습 방법**: 하나의 배치에서 $k$개의 서로 다른 비디오 샘플 $(X_i, Z_i, Y_i)$를 선택한다. 이때, 첫 번째 샘플의 타겟 패치 $Z_1$으로 생성한 템플릿 $\beta_1$ 하나만을 사용하여 $k$개의 서로 다른 검색 영역 $X_1, \dots, X_k$에서 타겟을 모두 찾아내도록 강제한다.
- **효과**: 초기 타겟 특징은 서로 다른 비디오에서 왔으므로 정렬되지(misaligned) 않았지만, Gradient 정보는 각 영역의 정답과 정렬되어 있다. 따라서 네트워크는 정렬되지 않은 초기 템플릿을 수정하기 위해 강제로 Gradient 정보에 집중하게 되며, 결과적으로 일반화 능력이 향상되고 과적합이 방지된다.

### 온라인 추적 (Online Tracking)
- **초기화**: 첫 프레임에서 위 과정을 통해 $\beta^*$와 $h_2(Z_1)$을 계산한다.
- **온라인 업데이트**: 추적 결과가 신뢰할 수 있다고 판단되는 샘플(스코어 맵의 최댓값이 임계값의 50% 이상인 경우)을 사용하여 5프레임마다 한 번씩 템플릿을 업데이트한다. 최종 템플릿은 초기 템플릿과 업데이트된 템플릿을 결합하여 사용한다.

## 📊 Results

### 실험 설정
- **데이터셋**: OTB-2015, TC-128, VOT-2017, LaSOT
- **지표**: Precision, Success Rate, EAO (Expected Average Overlap)
- **속도**: 80fps (Nvidia 1080ti GPU 기준)

### 주요 결과
1.  **OTB-2015**: Baseline인 SiameseFC 대비 Precision은 약 8%, Success Rate는 약 6% 향상되었다.
2.  **TC-128**: 비교 대상 트래커들 중 Precision과 Success 모두에서 최상위 성적을 거두었다.
3.  **VOT-2017**: EAO 기준 0.247을 기록하며, 실시간 챌린지 우승자인 CSRDCF++보다 3.5% 높은 성능을 보였다. 특히 훨씬 적은 학습 데이터(약 4,000개 비디오)를 사용하고도 SiamRPN보다 우수한 성능을 냈다.
4.  **LaSOT**: 대규모 데이터셋에서도 3위의 성적을 기록하였다. MDNet이나 VITAL보다 정확도는 낮으나, 속도 면에서 압도적(MDNet 1fps, VITAL 1.5fps $\rightarrow$ GradNet 80fps)인 우위를 점한다.

### Ablation Study
- Template Generalization을 제거한 모델(Ours w/o M)은 성능이 크게 하락하며, 이는 제안한 학습 방법이 Gradient 활용 능력을 높이는 데 필수적임을 입증한다.
- 업데이트 브랜치를 제거하거나($\text{w/o U}$) Gradient 적용을 제거한($\text{w/o MG}$) 경우 모두 baseline보다 낮거나 약간 높은 수준에 그쳐, GradNet 구조의 유효성이 확인되었다.

## 🧠 Insights & Discussion

### 강점
본 논문은 Siamese 네트워크의 고질적인 문제인 '온라인 적응력 부족'을 해결하기 위해, 전통적인 최적화 루프를 딥러닝 네트워크(GradNet)로 대체했다는 점에서 매우 혁신적이다. 특히 Meta-learning의 아이디어를 추적 문제에 맞게 변형하여 실시간성을 유지하면서도 온라인 업데이트가 가능하게 만든 점이 돋보인다.

### 한계 및 논의사항
- **신뢰성 판단의 임계값**: 온라인 업데이트 시 스코어 맵의 최댓값을 기준으로 신뢰성 있는 샘플을 선택하는데, 이 임계값 설정이 추적 성능에 영향을 줄 수 있다.
- **데이터 의존성**: Template Generalization이 성능 향상에 결정적인 역할을 하는데, 이는 학습 데이터의 구성과 $k$값(배치 내 샘플 수)에 따라 성능 편차가 발생할 가능성이 있다.
- **비판적 해석**: 2차 Gradient를 사용하는 학습 과정이 계산 복잡도를 높일 수 있으나, 추론 단계에서는 고정된 네트워크를 사용하므로 실시간성에 영향을 주지 않는 영리한 설계를 취했다.

## 📌 TL;DR

본 논문은 Siamese 기반 트래커의 실시간 속도를 유지하면서 온라인 적응력을 높이기 위해, **Gradient 정보를 이용해 템플릿을 업데이트하는 GradNet**을 제안한다. CNN을 통해 비선형 Gradient 최적화 과정을 모사함으로써 단 한 번의 iteration으로 템플릿을 갱신하며, **Template Generalization** 학습법을 통해 과적합을 방지하고 Gradient 활용 능력을 극대화하였다. 결과적으로 80fps의 빠른 속도로 VOT-2017 EAO 등 주요 벤치마크에서 SOTA 수준의 성능을 달성하였으며, 이는 향후 실시간 고정밀 추적 연구에 중요한 방향성을 제시한다.