# MULTISTAR: INSTANCE SEGMENTATION OF OVERLAPPING OBJECTS WITH STAR-CONVEX POLYGONS

Florin C. Walter, Sebastian Damrich, Fred A. Hamprecht (2021)

## 🧩 Problem to Solve

본 논문은 생의학 이미지(biomedical images)에서 서로 겹쳐져 있는 객체들의 인스턴스 분할(instance segmentation) 문제를 해결하고자 한다. 특히 현미경 이미지 내의 세포들과 같이 객체들이 밀집되어 있고 투영 과정에서 서로 겹쳐 보일 때, 기존의 분할 방법들은 어려움을 겪는다. 

이 문제의 핵심적인 어려움은 두 가지이다. 첫째, 객체가 겹치는 영역에서는 해당 픽셀이 어떤 객체에 속하는지 모호하기 때문에, 객체의 형태를 정의하는 파라미터(예: Star Distances)나 객체 존재 확률(Object Probability)을 정의하기 어렵다. 둘째, 일반적인 Non-Maximum Suppression(NMS) 과정에서는 두 제안(proposal)의 IoU(Intersection over Union)가 높을 경우 하나를 제거하는데, 실제로 객체가 겹쳐 있는 경우에는 서로 다른 두 객체임에도 불구하고 IoU가 높게 측정되어 실제 객체가 삭제되는 문제가 발생한다.

따라서 본 연구의 목표는 기존의 효율적인 인스턴스 분할 방법인 StarDist를 확장하여, 겹쳐진 객체들을 정확하게 분리하고 검출할 수 있는 MultiStar 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Overlap Probability($P_{over}$)**라는 새로운 예측 채널을 추가하여 겹침 정보를 명시적으로 활용하는 것이다.

구체적인 설계 직관은 다음과 같다. 픽셀 수준에서 객체의 겹침 여부를 예측하고, 이 정보를 이용해 (1) 겹치지 않는 영역에서만 제안(proposal)을 샘플링하여 파라미터 정의의 모호성을 해결하며, (2) NMS 단계에서 겹침 영역을 교집합 계산에서 제외함으로써 실제 겹쳐진 객체들이 서로를 억제하지 않도록 하는 것이다. 이를 통해 StarDist의 단순한 구조를 유지하면서도 겹쳐진 객체 분할 능력을 획기적으로 향상시켰다.

## 📎 Related Works

기존에 겹쳐진 객체를 분할하기 위한 여러 접근 방식이 존재하였다. 일부 연구들은 라벨 공간을 3D로 확장하여 마스크를 전단(shear)함으로써 겹치지 않는 표현으로 변환하거나, Encoder-Decoder 네트워크를 이용한 end-to-end 방식, 혹은 인스턴스 간의 관계 상호작용(instance relation interaction)을 이용하는 방식 등이 제안되었다. 또한 근원적으로 원형에 가까운 객체들을 위해 변분법(variational method)을 사용하는 연구도 있었다.

반면, StarDist는 Star-convex polygons를 이용하여 객체를 파라미터화함으로써 밀집된 객체 분할에서 매우 효율적인 성능을 보였으나, 기본적으로 객체가 겹치지 않는다는 가정을 전제로 한다. MultiStar는 이러한 StarDist의 우아한 파라미터화 방식을 유지하면서, 겹침 문제를 해결하기 위한 최소한의 수정만을 가했다는 점에서 기존 연구들과 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인
MultiStar는 UNet 백본을 기반으로 하며, 세 개의 출력 브랜치를 통해 각각 Object Probability, Star Distances, 그리고 Overlap Probability를 예측한다. 예측된 결과들은 이후 제안 샘플링 과정과 수정된 NMS 과정을 거쳐 최종 인스턴스 마스크로 변환된다.

### 주요 구성 요소 및 절차

**1. 제안 샘플링 (Proposal Sampling)**
기존 StarDist는 Object Probability($P_{obj}$)가 높은 지점에서 제안을 생성한다. 하지만 MultiStar는 겹침 영역에서 샘플링이 일어날 경우 파라미터 정의가 모호해지는 것을 방지하기 위해, 다음과 같이 수정된 샘플링 확률 $P_{proposal}$을 사용한다.

$$P_{proposal}(p) \propto P_{obj}(p) \cdot (1 - P_{over}(p))$$

이 식에 따라, 모델이 겹침 확률 $P_{over}$를 높게 예측한 픽셀에서는 샘플링 확률이 낮아지며, 결과적으로 파라미터화가 명확한 비겹침 영역에서만 제안이 생성된다.

**2. 수정된 NMS (Modified NMS)**
두 제안 $A$와 $B$가 겹칠 때, 이것이 동일 객체에 대한 중복 검출인지 아니면 실제 겹쳐진 두 객체인지를 판별하기 위해 교집합(Intersection) 계산 방식을 변경한다. 겹침 영역으로 예측된 부분은 교집합 계산에서 제외한다.

$$I \equiv \sum_{p \in A \cap B} (1 - P_{over}(p))$$

이렇게 계산된 $I$를 이용하여 IoU를 구하면, 실제로 겹쳐진 객체들 사이의 IoU는 낮게 측정되어 NMS에 의해 제거되지 않고 보존된다. 반면, 겹침이 예측되지 않은 영역에서의 중복은 여전히 높은 IoU를 가져 제거된다.

### 모델 아키텍처
- **Backbone**: 5개 레벨(16, 32, 64, 128, 256 채널)을 가진 generic UNet을 사용한다. 각 블록은 두 번의 $3 \times 3$ 합성곱, Batch Normalization, ReLU, 그리고 max-pooling 또는 upsampling으로 구성된다.
- **Output Branches**:
    - **Object Probability**: 단일 채널, Sigmoid 활성화 함수.
    - **Overlap Probability**: 단일 채널, Sigmoid 활성화 함수.
    - **Star Distances**: 32개 방향에 대한 32개 채널, ReLU 활성화 함수.

### 훈련 목표 및 손실 함수
네트워크 파라미터 $\theta$와 태스크별 불확실성(task uncertainties) $\sigma_i$를 함께 최적화하는 정규화된 가중 합 손실 함수를 사용한다.

$$L(\theta, \sigma_i) = \frac{1}{\sigma_{over}^2} L_{over}(\theta) + \frac{1}{\sigma_{obj}^2} L_{obj}(\theta) + \frac{1}{\sigma_{dist}^2} L_{dist}(\theta) + \log(\sigma_{over} \sigma_{obj} \sigma_{dist})$$

여기서 $L_{over}$와 $L_{obj}$는 Binary Cross-Entropy 손실을 사용하며, $L_{dist}$는 예측된 Star Distances와 실제 값 사이의 평균 절대 오차(Mean Absolute Difference)를 사용한다. 이때 $L_{dist}$의 각 픽셀 기여도는 실제 Object Probability로 가중치를 둔다. 특히, 실제 정답(Ground Truth)에서 객체가 겹치는 픽셀들은 $L_{obj}$와 $L_{dist}$ 계산에서 제외된다.

## 📊 Results

### 실험 설정
- **데이터셋**: 
    - **OSC-ISBI**: 자궁경부 세포 이미지 데이터셋.
    - **DSB-OV**: 기존의 비겹침 데이터셋(DSB 2018)에 객체를 무작위로 복제, 회전, 이동시켜 인위적으로 겹침(최소 15% 픽셀)을 생성한 합성 데이터셋.
- **지표**: Dice Coefficient (DC), Pixel-based True Positive($TP_p$) / False Positive($FP_p$) rates, Object-based False Negative($FN_o$) rate, 그리고 Average Precision (AP).

### 정량적 결과
- **OSC-ISBI 데이터셋**: MultiStar는 기존의 SOTA 방법들(Isoo DL, Diskmask)과 비교하여 DC, $FN_o$, $TP_p$ 측면에서 경쟁력 있는 수치를 보였다. 특히, 다른 방법들은 핵(nuclei) 어노테이션을 함께 사용했지만 MultiStar는 세포질(cytoplasm) 어노테이션만으로 학습했음에도 유사한 성능을 낸다는 점이 고무적이다.
- **DSB-OV 데이터셋**: 다양한 $\tau$ (IoU 임계값)에서 StarDist와 비교했을 때, MultiStar가 모든 구간에서 유의미하게 높은 AP를 기록하였다. 특히 StarDist는 겹쳐진 객체들을 하나로 합쳐버리는 경향이 있는 반면, MultiStar는 이를 효과적으로 분리해냈다.

### 정성적 결과
실험 결과, MultiStar는 겹쳐진 세포들의 경계를 명확히 구분하여 개별 인스턴스로 검출하는 능력이 StarDist보다 훨씬 뛰어남을 보여주었다.

## 🧠 Insights & Discussion

본 연구의 강점은 매우 단순한 구조적 변경(채널 추가 및 NMS 로직 수정)만으로 기존 StarDist의 적용 범위를 겹쳐진 객체 영역까지 확장했다는 점이다. 복잡한 3D 공간 투영이나 정교한 관계 네트워크 없이도 생의학 이미지의 특성을 잘 활용하여 실용적인 성능 향상을 이루어냈다.

다만, 한계점 또한 존재한다. MultiStar는 겹침 영역($P_{over}$가 높은 곳)에서 제안을 샘플링하지 않기 때문에, **다른 객체에 의해 완전히 가려진(fully overlapped) 객체는 검출할 수 없다.** 이는 본 모델이 픽셀 수준의 겹침 정보를 활용하는 방식의 근본적인 제약이다.

비판적으로 해석하자면, OSC-ISBI 데이터셋에서 DC와 같은 지표만으로 최적의 하이퍼파라미터를 정했을 때 실제 결과에서는 False Positive가 많이 발생하는 경향이 있었다. 이는 겹쳐진 객체 분할 문제에서 단순한 픽셀 일치도 지표보다는 AP와 같은 정밀도 기반 지표가 실제 활용성 판단에 더 중요함을 시사한다.

## 📌 TL;DR

MultiStar는 StarDist에 **Overlap Probability** 예측 기능을 추가하여 겹쳐진 객체들의 인스턴스 분할 문제를 해결한 모델이다. 겹침 정보를 이용해 샘플링 위치를 최적화하고 NMS의 교집합 계산 방식을 수정함으로써, 실제 겹쳐진 객체들이 서로를 억제하지 않고 개별적으로 검출되도록 하였다. 이 연구는 단순한 구조로도 겹침 문제를 효과적으로 해결할 수 있음을 보였으며, 향후 세포 분할과 같은 생의학 이미지 분석 분야에서 효율적인 베이스라인으로 활용될 가능성이 높다.