# Diversity-Promoting Ensemble for Medical Image Segmentation

Mariana-Iuliana Georgescu, Radu Tudor Ionescu, and Andreea-Iuliana Miron (2023)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 정확한 진단과 치료 계획 수립을 위해 매우 높은 정밀도가 요구되는 작업이다. 예를 들어, 악성 종양의 정밀한 분할은 방사선 치료 시 방사선 조사량의 정확한 보정으로 이어진다. 최근 딥러닝, 특히 U-Net 기반의 모델들이 널리 사용되고 있으나, 단일 신경망만으로는 최적의 성능을 달성하는 데 한계가 있다.

일반적으로 여러 모델을 결합하는 앙상블(Ensemble) 기법은 정확도를 높이는 검증된 방법이다. 그러나 단순히 성능이 좋은 상위 모델들만 선택하여 앙상블을 구성할 경우, 모델 간의 예측 결과가 서로 매우 유사하여(상관관계가 높아) 앙상블을 통한 성능 향상 효과가 제한적이라는 문제가 있다. 따라서 본 논문은 모델 간의 다양성(Diversity) 또는 비상관성(Decorrelation)을 확보함으로써 서로의 결정을 보완하고 분할 정확도를 극대화하는 앙상블 생성 전략을 제안하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 앙상블을 구성하는 모델들 사이의 출력값에 대한 상관관계를 측정하고, 이를 바탕으로 서로 다른 특성을 가진 모델들을 선택하여 다양성을 증진시키는 것이다. 

특히, 두 모델의 출력 결과물 사이의 Dice score를 사용하여 상관관계를 정의하며, 단순한 성능 지표뿐만 아니라 모델 간의 비상관성을 동시에 고려하는 하향식(Bottom-up) 선택 알고리즘인 **Diversity-Promoting Ensemble (DiPE)** 전략을 제안한다. 이를 통해 제한된 모델 수(Budget) 내에서 최적의 성능을 낼 수 있는 모델 조합을 효율적으로 구성할 수 있다.

## 📎 Related Works

의료 영상 분할 연구는 크게 2D 이미지 기반과 3D 볼륨 기반 접근 방식으로 나뉜다. 2D 분할에서는 U-Net이 가장 대표적인 아키텍처이며, 이를 개선한 mU-Net과 같은 변형 모델들이 제안되었다. 3D 분할에서는 VoxResNet과 같이 잔차 연결(Residual connection)을 활용하거나 공간 주의 집중(Spatial attention) 메커니즘을 도입하여 성능을 높이려는 시도가 있었다.

앙상블 관점에서는 각 평면(axial, sagittal, coronal)별로 모델을 사용해 융합하거나, 2D와 3D 모델을 혼합하여 사용하는 AdaEn-Net과 같은 연구가 존재한다. 하지만 기존의 앙상블 방식들은 모델들을 단순히 결합하거나 k-fold 교차 검증을 통해 생성된 모델들을 평균 내는 방식을 사용했을 뿐, 모델 출력 간의 상관관계를 직접적으로 계산하여 다양성을 확보하려는 시도는 부족했다. 본 논문은 이러한 상관관계 분석을 통한 모델 선택 과정을 도입함으로써 기존 연구와 차별점을 갖는다.

## 🛠️ Methodology

### 1. 신경망 아키텍처 구성
연구진은 다양성을 확보하기 위해 총 9가지의 서로 다른 U-Net 변형 모델을 생성하였다. 변동성은 다음 세 가지 요소에서 부여되었다.

- **Backbone Variations**: 인코더로 ResNet-34, EfficientNet-B0, EfficientNet-B1의 세 가지 구조를 사용하였다.
- **Attention Mechanism**: 의료 영상 특화 주의 집중 기법인 Multi-head convolutional attention (MHCA) 모듈을 스킵 연결(Skip connection)의 마지막 두 레이어에 추가하여 모델의 변동성을 높였다.
- **Loss Function**: 클래스 불균형 문제를 해결하기 위해 Binary Cross Entropy (BCE) 손실과 Tversky 손실을 결합하여 사용하였다.

전체 손실 함수 $L_{\text{total}}$은 다음과 같이 정의된다:
$$L_{\text{total}}(Y, \hat{Y}) = 0.5 \cdot L_{\text{BCE}}(Y, \hat{Y}) + 0.5 \cdot L_{\text{Tversky}}(Y, \hat{Y})$$

여기서 $L_{\text{BCE}}$는 다음과 같다:
$$L_{\text{BCE}}(Y, \hat{Y}) = -(\omega \cdot Y \cdot \log(\sigma(\hat{Y})) + (1-Y) \cdot \log(1-\sigma(\hat{Y})))$$
이때 $\omega$는 양성 클래스에 부여하는 가중치로, 표준 BCE에서는 $\omega=1$을 사용하며, 양성 클래스에 더 높은 가중치를 두어 False Negative를 줄이기 위해 $\omega=10$을 적용한 Positively-biased BCE를 함께 실험하였다.

Tversky 손실 함수는 다음과 같이 정의된다:
$$L_{\text{Tversky}}(Y, \hat{Y}) = \frac{\sum_{i=1}^{m} \hat{y}_{0,i} \cdot y_{0,i}}{\sum_{i=1}^{m} \hat{y}_{0,i} \cdot y_{0,i} + \alpha \cdot \sum_{i=1}^{m} \hat{y}_{0,i} \cdot y_{1,i} + \beta \cdot \sum_{i=1}^{m} \hat{y}_{1,i} \cdot y_{0,i}}$$
(단, $\alpha=0.5, \beta=0.5$ 사용)

### 2. Diversity-Promoting Ensemble (DiPE) 전략

#### 상관관계 행렬(Correlation Matrix) 정의
두 모델 $M_i$와 $M_j$의 출력 결과물 $\hat{Y}_{i,n}$과 $\hat{Y}_{j,n}$ 사이의 상관관계 $C_{i,j}$는 검증 세트 내 모든 샘플 $n$에 대한 Dice coefficient의 평균으로 계산한다.
$$\text{Dice}(\hat{Y}_{i,n}, \hat{Y}_{j,n}) = \frac{2 \cdot |\hat{Y}_{i,n} \cap \hat{Y}_{j,n}|}{|\hat{Y}_{i,n}| + |\hat{Y}_{j,n}|}$$
$$C_{i,j} = \frac{1}{t} \sum_{n=1}^{t} \text{Dice}(\hat{Y}_{i,n}, \hat{Y}_{j,n})$$

#### 앙상블 구성 알고리즘
제한된 모델 수 $k$를 가진 앙상블 $E$를 구축하는 절차는 다음과 같다.
1. 검증 세트에서 가장 성능이 좋은(Dice score가 높은) 단일 모델을 먼저 선택하여 $E$에 추가한다.
2. 아직 선택되지 않은 모델 $M_i$들에 대해, 현재 $E$에 포함된 모델들과의 평균 상관관계와 해당 모델의 성능 저하분(Dice error)을 합산하여 점수를 계산한다.
$$C_{i,E} = \text{avg}((1-d_i) + C_{i,j}), \quad \forall M_j \in E$$
여기서 $d_i$는 모델 $M_i$의 Ground-truth 대비 Dice score이다. $(1-d_i)$ 항은 모델의 절대적인 성능을 고려하기 위함이며, $C_{i,j}$ 항은 기존 모델들과의 다양성을 확보하기 위함이다.
3. $C_{i,E}$ 값이 최소가 되는 모델을 선택하여 $E$에 추가하며, 이 과정을 $k$개의 모델이 모일 때까지 반복한다.

최종 예측은 각 모델의 Softmax 활성화 값의 평균을 구하는 **Soft plurality voting** 방식을 통해 수행한다.

## 📊 Results

### 실험 설정
- **데이터셋**: UW-Madison GI Tract Image Segmentation 데이터셋 (위, 소장, 대장 분할)
- **평가 지표**: Intersection over Union (IoU) 및 Dice coefficient
- **구현**: PyTorch, ImageNet 사전 학습 가중치 사용, Adam 옵티마이저 적용

### 정량적 결과
- **단일 모델**: EfficientNet-B1 기반 모델이 가장 높은 성능(Dice 0.9130)을 보였으며, ResNet-34는 상대적으로 낮은 성능을 기록하였다. $\omega=10$을 적용한 Positively-biased BCE loss가 성능 향상에 기여하였다.
- **앙상블 비교**: 
    - 모델 수 $k \ge 3$인 경우, DiPE 전략이 상위 $k$개 모델을 선택한 Baseline 앙상블보다 일관되게 우수한 성능을 보였다.
    - 특히 $k=8$일 때의 DiPE 앙상블(Dice 0.9201)이 가용 가능한 모든 모델(9개)을 전부 사용한 앙상블(Dice 0.9199)보다 더 높은 성능을 기록하며, 적은 수의 다양한 모델 조합이 단순한 다수 모델의 결합보다 효과적임을 입증하였다.

### 절제 연구 (Ablation Study)
상관관계 계산 식(Eq. 7)에서 모델 성능 지표인 $(1-d_i)$를 제거했을 때 성능이 크게 하락하였다. 이는 단순히 다양성만 추구할 경우, 성능이 매우 낮은 모델이 선택될 위험이 있으며, 따라서 '모델의 성능'과 '상관관계'를 동시에 고려하는 것이 필수적임을 보여준다.

### 정성적 결과
시각화 결과, DiPE는 위(Stomach)와 횡행결장(Transverse colon)을 구분하는 능력이 Baseline보다 뛰어났으며, 특히 지방 조직이 많은 소장(Small bowel) 및 장간막(Mesentery) 영역에서 Ground-truth에 매우 근접한 정밀한 분할 결과를 생성하였다.

## 🧠 Insights & Discussion

본 연구는 앙상블 구성 시 단순히 개별 모델의 성능이 높은 것보다, 모델들이 서로 다른 오류를 범하는 '비상관성'을 갖는 것이 전체 시스템의 성능 향상에 더 결정적이라는 점을 시사한다.

**강점 및 해석:**
- 추가적인 학습 과정 없이 기존 모델들의 출력값 비교만으로 최적의 앙상블 조합을 찾을 수 있어 연산 효율적이다.
- 추론 단계에서 발생하는 오버헤드가 Baseline 앙상블과 동일하므로 실용성이 높다.
- Dice score라는 단순한 지표를 상관관계 측정 도구로 재정의하여 효과적으로 사용하였다.

**한계 및 향후 과제:**
- 본 연구는 Soft plurality voting(단순 평균)을 사용하였으나, 저자들은 향후 Meta-learner를 도입하여 가중치를 학습시킨다면 더 높은 성능을 기대할 수 있다고 언급하였다.
- 특정 데이터셋(GI Tract)에 한정된 실험이므로, 다양한 의료 영상 도메인으로의 확장 검증이 필요하다.

## 📌 TL;DR

이 논문은 의료 영상 분할에서 모델 간의 출력 상관관계를 Dice score로 측정하여, 성능과 다양성을 동시에 최적화하는 모델 선택 전략인 **DiPE (Diversity-Promoting Ensemble)**를 제안한다. 실험 결과, 단순히 성능 상위 모델을 모은 앙상블보다 DiPE를 통해 구성된 소수의 비상관 모델 집합이 더 높은 정확도를 보였으며, 이는 의료 영상 분할에서 모델 간의 다양성 확보가 필수적임을 입증한다. 이 연구는 향후 제한된 자원으로 고성능 의료 영상 분석 시스템을 구축하는 데 중요한 가이드라인을 제공한다.