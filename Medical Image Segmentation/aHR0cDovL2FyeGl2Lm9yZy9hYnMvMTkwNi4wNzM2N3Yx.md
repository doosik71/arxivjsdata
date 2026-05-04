# A sparse annotation strategy based on attention-guided active learning for 3D medical image segmentation

Zhenxi Zhang, Jie Li, Zhusi Zhong, Zhicheng Jiao, and Xinbo Gao (Year not provided)

## 🧩 Problem to Solve

본 논문은 3D 의료 영상 분할(3D medical image segmentation)에서 발생하는 **어노테이션 비용의 과다 문제**를 해결하고자 한다. 3D 의료 영상 데이터셋을 구축하기 위해서는 해부학적 전문가의 정밀한 라벨링이 필수적이며, 이는 매우 많은 인력과 시간을 소모하는 단조롭고 고비용의 작업이다.

기존의 딥러닝 기반 분할 모델들은 대부분 전체 데이터가 라벨링된 Full annotated dataset을 필요로 한다. 이를 완화하기 위해 일부 슬라이스만 라벨링하는 Sparse annotation 방식이 제안되었으나, 어떤 슬라이스가 모델 학습에 가장 유용한 정보(Informative)를 담고 있는지에 대한 효율적인 선택 전략이 부족한 상태였다. 따라서 본 연구의 목표는 **Attention mechanism을 활용하여 라벨링이 필요한 최적의 슬라이스를 선택하는 Active Learning(AL) 전략을 제안함으로써, 최소한의 라벨링만으로 전체 라벨링과 유사한 성능을 달성하는 것**이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Attention Map(AM)의 신뢰도가 실제 분할 정확도와 높은 상관관계를 가진다**는 점을 이용하는 것이다. 구체적인 기여 사항은 다음과 같다.

1. **Attention-guided Active Learning 전략 제안**: 3D 의료 영상 분할을 위해 Attention mechanism을 이용한 효율적인 suggestive annotation 전략을 구축하였다.
2. **Attention-embedded 3D U-Net 설계**: Channel Attention Mechanism(CAM)과 Spatial Attention Mechanism(SAM)을 3D U-Net에 결합하여 분할 정확도를 높이는 동시에, 학습된 모델의 상태를 피드백하여 라벨링할 슬라이스를 결정하는 지표로 활용하였다.
3. **새로운 불확실성 측정 지표 도입**: Pseudo Dice Similarity Coefficient(P-DSC)와 Pseudo accuracy(P-accuracy)를 제안하여, Ground Truth(GT) 없이도 라벨링되지 않은 슬라이스의 분할 품질을 정확하게 예측하였다.
4. **라벨링 비용의 획기적 절감**: 실험을 통해 뇌 추출(Brain extraction) 작업에서는 15%~20%, 조직 분할(Tissue segmentation) 작업에서는 30%~35%의 슬라이스만 라벨링하고도 Full annotation과 대등한 성능을 얻을 수 있음을 입증하였다.

## 📎 Related Works

본 논문에서는 다음과 같은 관련 연구들을 언급하며 차별점을 제시한다.

- **3D Segmentation 모델**: U-Net의 확장형인 3D U-Net과 V-Net이 표준적으로 사용되고 있다. 하지만 이들은 특징 맵(Feature Maps, FMs) 내의 모든 정보가 동일하게 중요하다고 가정하는 경향이 있다.
- **Attention Mechanism**: Squeeze-and-Excitation(SE) block과 같은 기법들이 채널별 특징 맵을 재보정하여 성능을 향상시킨 사례가 있다.
- **Active Learning (AL)**: AIFT와 같은 프레임워크가 엔트로피(Entropy)나 다양성(Diversity), 혹은 모델의 불확실성(Uncertainty)과 이미지 간 유사성을 기준으로 샘플을 선택하는 방식을 사용해 왔다.

**기존 방식과의 차별점**: 기존의 AL 전략들은 불확실성과 유사성을 측정하지만, 이것이 최종 평가 지표(Evaluation criteria)와 직접적으로 연결되지 않는다는 한계가 있다. 반면, 본 논문은 Attention Map을 통해 실제 분할 정확도와 직접적으로 연관된 지표(P-DSC, P-accuracy)를 산출하여 더 정교한 샘플 선택이 가능하게 하였다.

## 🛠️ Methodology

### 1. Attention-embedded 3D U-Net

본 모델은 3D U-Net 구조를 기본으로 하며, Encoder와 Decoder 사이의 Skip connection 및 Decoder 서브모듈에 Attention mechanism을 추가하였다.

- **CAM (Channel Attention Mechanism)**: 각 채널의 중요도를 학습하여 자동으로 획득하고 채널 방향의 특징 맵을 재보정한다.
- **SAM (Spatial Attention Mechanism)**: 특징 맵의 공간적 스케일에서 가중치를 부여하여 모델이 중요한 특징 영역(Salient regions)에 집중하도록 한다.
- **결합 방식**: CAM과 SAM을 병렬로 배치하여 얻은 결합 Attention Map을 입력 특징 맵에 곱함으로써 유의미한 특징은 강조하고 불필요한 특징은 억제한다.

### 2. Attention-guided Active Learning 절차

AL 프로세스는 다음과 같은 반복적인 루프(Iterative loop)로 구성된다.

1. **초기 학습**: axial plane 기준 16 또는 32 슬라이스 간격으로 샘플링된 초기 데이터셋으로 모델을 학습시킨다.
2. **예측 및 분석**: 학습된 모델에 데이터를 입력하여 최종 특징 맵(Final FMs)과 Attention Map(AMs)을 얻는다.
    - $SR_1$: Final FMs를 통한 분할 결과
    - $SR_2$: AMs를 통한 분할 결과
3. **불확실성 측정 (P-DSC & P-accuracy)**:
    - **P-DSC**: $SR_1$과 $SR_2$ 사이의 Dice Similarity Coefficient를 계산하여 슬라이스별 품질을 예측한다.
    - **P-accuracy**: 다중 클래스 분할(Tissue segmentation)의 경우, 배경(class 0)을 제외하고 $SR_1$과 $SR_2$가 동일하게 예측한 픽셀의 비율을 계산한다.
4. **슬라이스 선택**: P-DSC 또는 P-accuracy 값이 낮은(즉, $SR_1$과 $SR_2$의 차이가 커서 불확실성이 높은) 슬라이스를 선택하여 전문가에게 라벨링을 요청한다.
5. **업데이트 및 재학습**: 새로 라벨링된 데이터를 학습 세트에 추가하고 모델을 Fine-tuning 한다. 이 과정은 전체 슬라이스의 평균 P-DSC/P-accuracy 변화량이 임계값 $\sigma = 0.005$보다 작아질 때까지 반복된다.

### 3. 주요 방정식

실제 평가 지표(Real)와 제안된 가상 지표(Pseudo)의 정의는 다음과 같다.

- **Real DSC (R-DSC)**:
    $$\text{R-DSC}[i] = \frac{2|SR_1[i] \cap GT[i]|}{|SR_1[i]| + |GT[i]|}$$
- **Pseudo DSC (P-DSC)**:
    $$\text{P-DSC}[i] = \frac{2|SR_1[i] \cap SR_2[i]|}{|SR_1[i]| + |SR_2[i]|}$$
- **Real Accuracy (R-accuracy)**:
    $$\text{R-accuracy}[i] = \frac{2|SR_1[i] \cap GT[i]|}{|SR_1[i]_{\neq 0}| + |GT[i]_{\neq 0}|}$$
- **Pseudo Accuracy (P-accuracy)**:
    $$\text{P-accuracy}[i] = \frac{2|SR_1[i] \cap SR_2[i]|}{|SR_1[i]_{\neq 0}| + |SR_2[i]_{\neq 0}|}$$
    *(여기서 $\neq 0$은 배경 클래스를 제외함을 의미한다.)*

## 📊 Results

### 1. 실험 설정

- **데이터셋**: dHCP (Developing Human Connectome Project)의 신생아 뇌 MRI 데이터 40개 샘플.
- **작업(Task)**: Task 1(뇌 추출, Brain extraction), Task 2(조직 분할, Tissue segmentation).
- **구현**: PyTorch, SGD Optimizer ($\text{lr}=0.001$), NVIDIA GTX 1080Ti GPU 2대 사용.
- **손실 함수**: Weighted cross-entropy loss를 사용하며, 라벨링되지 않은 슬라이스의 가중치는 0으로 설정하였다.

### 2. 정량적 결과

- **모델 성능 비교**: Full annotated dataset을 사용했을 때, 제안 방법은 3D U-Net 대비 Task 2에서 약 2.4% 향상된 F1 score(0.872)를 기록하였다.
- **AL 전략 비교**: Random query, Equal-interval query, Uncertainty query와 비교했을 때, 제안한 Attention-guided AL이 가장 적은 라벨링 양으로 높은 성능을 보였다.
- **라벨링 비용 효율성**:
  - **Task 1**: 전체의 **15%~20%** 슬라이스만 라벨링해도 Full annotation 수준의 성능에 도달하였다.
  - **Task 2**: 전체의 **30%~35%** 슬라이스 라벨링 시 Full annotation 수준의 성능을 달성하였다.

### 3. 지표의 유효성

P-DSC와 P-accuracy가 실제 정확도(R-DSC, R-accuracy)와 매우 강한 양의 상관관계를 가짐을 확인하였다. 선형 회귀 분석 결과, 회귀 계수가 각각 0.93(DSC)과 0.92(Accuracy)로 나타나, AM을 통한 품질 예측이 매우 정확함을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 Attention mechanism이 단순히 성능 향상을 위한 도구를 넘어, 모델이 스스로 자신의 예측 성능을 진단하는 '자기 감시(Self-monitoring)' 도구로 활용될 수 있음을 보여주었다. 특히, Attention Map의 분포가 학습이 진행됨에 따라 Ground Truth의 분포와 유사해진다는 점을 이용해 AL의 가이드를 제공한 점이 매우 독창적이다.

**강점**:

- 전통적인 AL의 불확실성 측정 방식(엔트로피 등)보다 의료 영상의 특성에 맞는 직접적인 성능 지표(P-DSC)를 설계하여 효율성을 높였다.
- CAM과 SAM을 동시에 적용하여 채널과 공간 정보를 모두 활용한 점이 유효하였다.

**한계 및 논의**:

- **전문가 라벨링 시뮬레이션**: 실험에서 전문가의 라벨링 과정을 '마스크를 제거하고 원래 어노테이션으로 되돌리는 방식'으로 시뮬레이션하였다. 이는 실제 임상 환경에서 전문가가 처음부터 라벨링하는 상황과는 차이가 있을 수 있다.
- **데이터셋 규모**: 40명의 신생아 데이터라는 상대적으로 작은 규모의 데이터셋에서 실험이 진행되었으므로, 더 대규모의 다양한 데이터셋에서의 일반화 성능 검증이 필요하다.

## 📌 TL;DR

본 연구는 3D 의료 영상 분할의 고비용 라벨링 문제를 해결하기 위해 **Attention-guided Active Learning** 전략을 제안하였다. 3D U-Net에 CAM과 SAM을 결합하여 Attention Map을 생성하고, 이를 통해 계산된 **P-DSC 및 P-accuracy 지표로 라벨링이 시급한 최적의 슬라이스를 선택**한다. 그 결과, 뇌 추출 작업에서는 15~20%, 조직 분할 작업에서는 30~35%의 적은 라벨링만으로도 전체 라벨링과 대등한 성능을 확보하여 어노테이션 비용을 획기적으로 줄였다. 이 연구는 향후 데이터 획득 비용이 높은 다양한 3D 의료 영상 분석 분야에 응용될 가능성이 높다.
