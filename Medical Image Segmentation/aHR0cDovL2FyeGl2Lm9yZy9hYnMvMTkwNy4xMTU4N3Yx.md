# Self-Adaptive 2D-3D Ensemble of Fully Convolutional Networks for Medical Image Segmentation

Maria G. Baldeon Calisto, Susana K. Lai-Yuen (2019)

## 🧩 Problem to Solve

의료 영상 분석에서 Segmentation은 매우 중요한 단계이다. 최근 Fully Convolutional Networks (FCNs)가 다양한 의료 영상 데이터셋에서 뛰어난 성과를 보이고 있으나, 다음과 같은 몇 가지 핵심적인 문제점이 존재한다.

첫째, 네트워크 아키텍처 설계가 대부분 특정 작업에 맞춰 수동으로 이루어진다. 이로 인해 새로운 데이터셋에 모델을 적용하려면 전문가의 광범위한 지식과 많은 시간이 소요되며, 이는 사실상 블랙박스 최적화 과정과 다름없다.

둘째, 의료 영상의 특성상 대용량의 volumetric data를 처리해야 하므로 아키텍처가 매우 크고 복잡해지는 경향이 있다. 기존의 2D FCN은 슬라이스 내 정보(intra-slice)는 잘 포착하지만 z-축 방향의 공간적 상관관계(inter-slice)를 활용하지 못하며, 3D FCN은 volumetric 정보를 직접 처리할 수 있지만 파라미터 수가 급격히 증가하여 높은 계산 비용과 GPU 메모리를 요구한다.

셋째, 기존의 Neural Architecture Search (NAS) 연구들이 주로 이미지 분류나 언어 모델링에 집중되어 있으며, 의료 영상 segmentation에 적용된 사례는 제한적이다. 또한 기존의 자동 설계 방식들은 volumetric 정보를 충분히 고려하지 않거나 네트워크의 크기를 최적화하는 부분에 소홀했다.

따라서 본 논문의 목표는 volumetric 정보를 통합하면서도 모델의 성능과 크기를 동시에 최적화할 수 있는 자가 적응형(self-adaptive) 2D-3D FCN 앙상블 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 서로 다른 특성을 가진 2D FCN과 3D FCN을 결합한 앙상블 구조를 설계하고, 이를 다목적 진화 알고리즘(Multiobjective Evolutionary Algorithm, MEA)을 통해 자동으로 최적화하는 것이다.

구체적으로, 2D FCN은 각 슬라이스의 평면 내 정보를 추출하고, 3D FCN은 슬라이스 간의 볼륨 정보를 활용하도록 하여 두 모델의 장점을 모두 취한다. 특히 단순히 성능(정확도)만 높이는 것이 아니라, 모델의 파라미터 수를 함께 최소화하는 다목적 최적화를 수행함으로써 효율적이고 가벼운 네트워크를 자동으로 설계한다는 점이 핵심적인 기여이다.

## 📎 Related Works

논문에서는 의료 영상 segmentation을 위한 기존 FCN 접근 방식을 두 가지 유형으로 구분하여 설명한다. 2D 네트워크 기반 방식은 이미지를 2D로 세그멘테이션한 후 이를 다시 결합하여 3D 결과를 얻지만, z-축의 공간적 상관관계를 활용하지 못하는 한계가 있다. 반면 3D FCN 방식은 3D 컨볼루션을 통해 volumetric 정보를 직접 처리하지만, 막대한 파라미터 수와 계산 시간이 필요하다는 단점이 있다. 이를 해결하기 위해 2D-3D 하이브리드 구조가 제안되었으나, 여전히 모델의 크기가 매우 크다는 문제가 남아 있다.

또한 NAS(Neural Architecture Search) 관련 연구들이 소개된다. 강화학습, 진화 알고리즘, 미분 가능한 탐색(DARTS 등) 방식이 존재하지만, 의료 영상 분야에서는 U-Net의 하이퍼파라미터를 조정하거나 고정된 템플릿 내에서 구성 요소를 선택하는 수준에 머물러 있었다. 본 논문은 기존의 자동 설계 방식들이 네트워크 크기를 최적화하지 못했다는 점을 지적하며, 다목적 최적화를 통해 성능과 크기의 Trade-off를 해결하고자 한다.

## 🛠️ Methodology

### 전체 파이프라인
제안된 시스템은 크게 두 단계(Phase)로 구성된다.
1. **Phase I (Architecture Adaptation):** 데이터셋을 5-fold로 나누고, 그중 무작위로 선택된 한 개의 폴드를 사용하여 2D FCN과 3D FCN의 최적 아키텍처를 탐색한다. 이때 MEA 알고리즘이 사용된다.
2. **Phase II (Ensemble Training):** Phase I에서 결정된 최적 아키텍처를 사용하여 5-fold 전체 데이터에 대해 각각 모델을 학습시킨다. 각 폴드당 2D FCN과 3D FCN의 예측 결과(softmax probability maps)를 평균 내고, 최종적으로 5개의 앙상블 모델 결과에 대해 다수결 투표(majority voting)를 실시하여 최종 세그멘테이션 맵을 생성한다.

### 네트워크 아키텍처
2D 및 3D FCN 모두 대칭적인 Encoder-Decoder 구조를 가지며, 기본 단위는 Residual Block이다.
- **Residual Block:** 세 개의 컨볼루션 단계(Conv $\rightarrow$ Batch Norm $\rightarrow$ Activation)로 구성되며, 첫 번째와 마지막 단계가 residual connection으로 연결되어 그래디언트 흐름을 개선한다.
- **Encoder:** 각 Residual Block 이후 stride 2의 Max-pooling을 적용하여 특성 맵의 크기를 절반으로 줄인다.
- **Decoder:** Transpose Convolution을 사용하여 특성 맵의 크기를 두 배로 늘린다.
- **Merge Operation:** Encoder와 Decoder의 대칭되는 지점 사이에서 low-level feature를 보존하기 위해 합산(Summation) 또는 연결(Concatenation) 작업을 수행한다.
- **기타:** 과적합 방지를 위해 Residual Block 이전에 Spatial Dropout을 적용하며, 최종 층은 $1 \times 1$ 커널과 Softmax 함수를 사용한다.

### 하이퍼파라미터 탐색 공간
MEA 알고리즘은 총 9개의 결정 변수를 최적화한다:
1. 총 Residual Block 수
2. 첫 번째 블록의 필터 수 (이후 pooling/transpose conv에 따라 2배 증가/절반 감소 규칙 적용)
3. $\sim$ 5. 각 컨볼루션 층의 커널 크기
6. 활성화 함수 (ReLU, ELU)
7. Merge 연산 방식 (Summation, Concatenation)
8. Spatial Dropout 확률
9. 학습률 (Learning Rate)

### 최적화 목표 및 방정식
MEA 알고리즘은 다음 두 가지 목적 함수를 동시에 최소화하는 Pareto Frontier를 찾는다.

첫 번째 목적 함수 $f_1$은 기대 세그멘테이션 에러를 최소화하는 것이며, Dice Similarity Coefficient (DSC)를 기반으로 정의된다. DSC의 정의는 다음과 같다.
$$DSC = \frac{2 \sum y_{pred} y_{gt}}{\sum y_{pred} + \sum y_{gt}}$$
여기서 $y_{pred}$는 예측된 복셀, $y_{gt}$는 실제 정답(ground truth) 복셀이다.

최종적인 $f_1$은 학습 세트와 검증 세트의 DSC, 그리고 모델이 충분히 학습되지 않았을 경우를 대비한 기대 개선량을 합산하여 계산한다.
$$Minimize \quad f_1(x) = \alpha(1 - DSC_{train}(x)) + \alpha(1 - DSC_{val}(x)) + \beta \left( \frac{E - e_{max}}{E} \right)$$
($\alpha, \beta$는 가중치 파라미터, $E$는 최대 에포크, $e_{max}$는 최대 검증 DSC가 나타난 에포크이다.)

두 번째 목적 함수 $f_2$는 모델의 크기를 최소화하는 것이다.
$$Minimize \quad f_2(x) = |\theta|$$
($|\theta|$는 네트워크 파라미터의 총 개수이다.)

## 📊 Results

### 실험 설정
- **데이터셋:** PROMISE12 전립선 MRI 이미지 세그멘테이션 챌린지 데이터 (학습 50개, 테스트 30개).
- **전처리:** $1 \times 1 \times 1.5\text{mm}$ 해상도로 리샘플링, $128 \times 128 \times 64$ 크기로 고정, 픽셀 강도는 표준편차 3배수 내로 clipping 후 0-1 범위로 스케일링.
- **입력 크기:** 2D FCN은 $128 \times 128$ 슬라이스, 3D FCN은 $96 \times 96 \times 16$ 패치 사용.

### 정량적 결과 및 비교
PROMISE12 챌린지의 온라인 리더보드에 제출한 결과, 제안된 2D-3D FCN 모델은 **전체 297개 제출물 중 9위**를 기록하였다.

- **성능 비교:** 수동으로 설계된 최상위 모델들(HD_Net, Bowda-Net 등)과 비교했을 때 DSC, 95% Hausdorff Distance (95 HD), Average Boundary Distance (ABD) 등 주요 지표에서 매우 근접한 성능을 보였다.
- **자동 설계 모델 비교:** 또 다른 자동 설계 프레임워크인 nnU-Net(I, II)보다 높은 순위를 기록하였다.
- **모델 크기:** 특히 모델 크기 면에서 압도적인 효율성을 보였다. nnU-Net의 경우 2D/3D 모델의 파라미터가 각각 $29.4 \times 10^6$ 및 $43.7 \times 10^6$개인 반면, 제안된 모델의 최적 아키텍처는 2D FCN이 $1.6 \times 10^6$개, 3D FCN이 $3.9 \times 10^6$개에 불과하였다.

### 정성적 결과
검증 세트에 대한 결과 확인 시, 별도의 Shape Prior나 특정 도메인 지식을 적용하지 않았음에도 불구하고 공간적으로 일관된 형태와 매끄러운 경계를 가진 세그멘테이션 결과를 생성하였다.

## 🧠 Insights & Discussion

본 연구의 가장 큰 강점은 **성능과 효율성 사이의 최적 균형점을 자동으로 찾았다**는 점이다. 수동 설계 모델들이 전립선과 주변 조직 간의 모호한 경계를 찾기 위해 복잡하고 특수한 메커니즘을 도입한 것과 달리, 본 모델은 단순한 FCN 구조의 앙상블과 다목적 최적화만으로 유사한 성능을 달성하였다.

특히 nnU-Net과 같은 강력한 자동화 프레임워크보다 훨씬 작은 파라미터 수로 더 나은 성능을 보였다는 점은, 단순히 규칙 기반의 적응(rule-based adaptation)보다 진화 알고리즘 기반의 다목적 최적화가 특정 데이터셋에 더 효율적인 아키텍처를 찾는 데 유효할 수 있음을 시사한다.

다만, 본 논문에서는 전립선 데이터셋 하나에 대해서만 검증을 수행하였으므로, 다른 장기나 다른 모달리티의 의료 영상에서도 동일하게 효율적인 아키텍처 탐색이 가능한지에 대한 추가 검증이 필요하다. 또한 진화 알고리즘 특성상 아키텍처 탐색 시간(2D 약 66시간, 3D 약 118시간)이 상당하다는 점이 실용적인 한계로 작용할 수 있다.

## 📌 TL;DR

본 논문은 의료 영상 세그멘테이션을 위해 **2D FCN(평면 정보)과 3D FCN(볼륨 정보)을 결합한 자가 적응형 앙상블 모델**을 제안한다. 다목적 진화 알고리즘(MEA)을 통해 **정확도 최대화와 파라미터 수 최소화**를 동시에 달성하는 최적의 아키텍처를 자동으로 설계하며, 이를 통해 PROMISE12 챌린지에서 매우 작은 모델 크기로도 상위 10위권(9위) 내의 경쟁력 있는 성능을 입증하였다. 이 연구는 향후 의료 영상 분석에서 전문가의 수동 설계 없이도 효율적이고 고성능인 맞춤형 네트워크를 구축하는 데 중요한 가능성을 제시한다.