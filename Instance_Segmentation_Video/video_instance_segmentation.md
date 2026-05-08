# Video Instance Segmentation

Linjie Yang, Yuchen Fan, Ning Xu (2019)

## 🧩 Problem to Solve

본 논문은 이미지 영역에서의 인스턴스 세그멘테이션(Instance Segmentation) 문제를 비디오 영역으로 확장한 새로운 컴퓨터 비전 태스크인 **Video Instance Segmentation (VIS)**을 정의하고 해결하고자 한다.

비디오 인스턴스 세그멘테이션의 목표는 비디오 내의 객체 인스턴스들에 대해 **탐지(Detection), 세그멘테이션(Segmentation), 그리고 추적(Tracking)**을 동시에 수행하는 것이다. 즉, 각 프레임에서 객체의 정밀한 마스크를 생성함과 동시에, 서로 다른 프레임 간에 동일한 객체 인스턴스를 식별하여 일관된 ID를 부여해야 한다.

이 문제의 중요성은 비디오 편집, 자율 주행, 증강 현실(AR)과 같이 비디오 레벨의 객체 마스크가 필요한 실제 응용 분야에서 매우 크다. 기존의 비디오 객체 세그멘테이션(VOS)은 객체의 카테고리를 인식할 필요가 없거나, 일부 데이터셋은 인스턴스 ID 정보가 부족하여 이 새로운 태스크를 연구하기 위한 대규모 벤치마크와 알고리즘이 절실한 상황이었다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **Video Instance Segmentation 태스크 정의**: 비디오 내 인스턴스의 탐지, 세그멘테이션, 추적을 통합한 새로운 문제를 공식적으로 정의하였다.
2. **YouTube-VIS 데이터셋 구축**: 2,883개의 고해상도 유튜브 비디오, 40개의 카테고리, 131k 개의 고품질 인스턴스 마스크를 포함하는 대규모 벤치마크 데이터셋을 제안하였다.
3. **MaskTrack R-CNN 알고리즘 제안**: Mask R-CNN을 기반으로 추적 브랜치(Tracking Branch)와 외부 메모리(External Memory)를 추가하여, 단일 네트워크에서 탐지, 세그멘테이션, 추적을 동시에 수행하는 모델을 설계하였다.

## 📎 Related Works

논문에서는 VIS와 관련된 기존 연구들을 다음과 같이 구분하여 설명한다.

* **Image Instance Segmentation**: 개별 이미지에서 객체를 탐지하고 마스크를 생성하지만, 프레임 간의 대응 관계(Correspondence)를 결정하는 추적 기능이 없다.
* **Video Object Tracking**: 주로 bounding box 기반의 추적(Detection-based Tracking)에 집중하며, 픽셀 단위의 세그멘테이션 마스크를 생성하지 않는다.
* **Video Object Detection**: 비디오 내 객체를 탐지하고 추적하지만, 세그멘테이션을 다루지 않는다.
* **Video Semantic Segmentation**: 픽셀 단위로 클래스를 분류하지만, 동일 클래스 내의 개별 인스턴스를 구분하거나 추적하지 않는다.
* **Video Object Segmentation (VOS)**: 객체를 세그멘테이션하고 추적하지만, 객체의 세만틱 카테고리(Semantic Category)를 인식하는 과정이 제외되어 있다.

결과적으로 VIS는 위 모든 요소(탐지, 세그멘테이션, 추적, 인식)를 통합해야 한다는 점에서 기존 연구들과 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인

MaskTrack R-CNN은 기존 Mask R-CNN의 구조(Classification, Bounding Box Regression, Mask Generation)에 **추적 브랜치(Tracking Branch)**와 **외부 메모리(External Memory)**를 추가한 형태이다. 비디오 프레임을 순차적으로 처리하는 온라인(Online) 방식으로 추론이 진행된다.

### 핵심 구성 요소 및 작동 원리

#### 1. New Tracking Branch

추적 브랜치는 각 후보 영역(Candidate Box)에 대해 인스턴스 레이블을 할당하는 역할을 한다. 이전 프레임들에서 이미 식별된 $N$개의 인스턴스가 있다면, 현재 후보 영역이 기존 인스턴스 중 하나인지, 아니면 새로운 인스턴스(레이블 0)인지를 결정하는 다중 클래스 분류 문제로 정의한다.

추적 확률 $p_i(n)$은 후보 영역의 특징 벡터 $f_i$와 메모리에 저장된 기존 인스턴스의 특징 벡터 $f_j$ 간의 내적(Dot product)을 통해 계산된다.

$$
p_i(n) =
\begin{cases}
\frac{e^{f_i^T f_n}}{1 + \sum_{j=1}^N e^{f_i^T f_j}}, & n \in [1, N] \\
\frac{1}{1 + \sum_{j=1}^N e^{f_i^T f_j}}, & n = 0
\end{cases}
$$

#### 2. External Memory 및 학습 절차

외부 메모리는 식별된 인스턴스들의 특징 벡터를 저장한다.

* **업데이트**: 후보 영역이 기존 인스턴스로 판명되면 최신 특징으로 메모리를 갱신하고, 새로운 인스턴스로 판명되면 메모리에 추가한다.
* **학습**: 두 개의 프레임(Reference frame, Query frame)을 무작위로 샘플링하여 학습한다. Reference frame에서는 GT(Ground Truth) 영역의 특징을 추출해 메모리에 넣고, Query frame에서는 후보 영역들이 메모리의 레이블과 잘 매칭되는지 Cross Entropy Loss를 통해 학습한다.
* **전체 손실 함수**:
    $$L = L_{cls} + L_{box} + L_{mask} + L_{track}$$

#### 3. 추적 정보의 결합 (Post-processing)

외형 유사도뿐만 아니라 세만틱 일관성, 공간적 상관관계, 탐지 신뢰도를 모두 활용하기 위해 추론 단계에서 다음과 같은 결합 점수 $v_i(n)$를 사용한다.

$$v_i(n) = \log p_i(n) + \alpha \log s_i + \beta \text{IoU}(b_i, b_n) + \gamma \delta(c_i, c_n)$$

여기서 $s_i$는 탐지 점수, $\text{IoU}(b_i, b_n)$은 bounding box 간의 겹침 정도, $\delta(c_i, c_n)$은 카테고리 일치 여부를 나타내는 크로네커 델타 함수이며, $\alpha, \beta, \gamma$는 하이퍼파라미터이다.

## 📊 Results

### 실험 설정

* **데이터셋**: YouTube-VIS (학습 2,238개, 검증 302개, 테스트 343개 비디오).
* **구현**: ResNet-50-FPN 백본 사용, NVIDIA 1080Ti GPU에서 20 FPS로 동작.
* **평가 지표**: AP(Average Precision)와 AR(Average Recall). 비디오 인스턴스 간의 시공간적 일관성을 측정하기 위해 다음과 같이 수정된 IoU를 사용한다.
    $$\text{IoU}(i,j) = \frac{\sum_{t=1}^T |m_i^t \cap \tilde{m}_j^t|}{\sum_{t=1}^T |m_i^t \cup \tilde{m}_j^t|}$$

### 주요 결과

MaskTrack R-CNN은 모든 지표에서 기존 베이스라인들을 상회하는 성능을 보였다.

| Methods | Val AP | Val $AP_{50}$ | Val $AP_{75}$ | Test AP |
| :--- | :---: | :---: | :---: | :---: |
| FEELVOS (Mask propagation) | 26.9 | 42.0 | 29.7 | 29.6 |
| DeepSORT (Track-by-detect) | 26.1 | 42.9 | 26.1 | 27.2 |
| SeqTracker (Offline) | 27.5 | 45.7 | 28.7 | 29.5 |
| **MaskTrack R-CNN** | **30.3** | **51.1** | **32.6** | **32.3** |

### 분석 및 통찰

1. **Ablation Study**: 결합 점수 식에서 Bounding Box IoU와 카테고리 일관성($\text{Cat}$)이 성능 향상에 가장 크게 기여함을 확인하였다. 이 두 요소가 없을 때 AP가 약 5% 하락하였다.
2. **Oracle 실험**:
    * 이미지 레벨의 정답을 제공했을 때(Image Oracle) 성능이 비약적으로 상승($AP \approx 78.7$)하였다.
    * 반면, 추적 정답을 제공했을 때(Identity Oracle)는 성능 향상이 미미했다.
    * 이는 현재 VIS 성능의 주된 병목 현상이 추적 알고리즘보다는 **이미지 레벨의 정밀한 탐지 및 세그멘테이션 성능**에 있음을 시사한다.

## 🧠 Insights & Discussion

### 강점

본 논문은 VIS라는 새로운 태스크를 정의하고 이를 위한 대규모 데이터셋을 구축함으로써 후속 연구의 기반을 마련하였다. 특히 Mask R-CNN에 추적 브랜치를 통합하여 end-to-end로 학습시킨 점이 효율적이며, 외부 메모리를 통해 객체가 일시적으로 사라졌다가 다시 나타나는 경우(Occlusion)에도 대응할 수 있는 능력을 갖추었다.

### 한계 및 비판적 해석

1. **단순한 결합 방식**: 추적 브랜치에서 나온 결과와 다른 큐(Cue)들을 단순 합산하는 포스트 프로세싱 방식으로 결합하였다. 논문에서도 언급했듯이, 이를 네트워크 내부에 통합하여 학습시키는 방향으로 개선될 여지가 크다.
2. **이미지 기반 특징 의존**: Oracle 실험 결과에서 드러났듯, 프레임 간의 시공간적 특징(Spatial-temporal features)을 충분히 활용하지 못하고 개별 프레임의 특징에 의존하고 있다. 이는 비디오 데이터가 가진 시간적 연속성이라는 이점을 완전히 활용하지 못한 결과라고 볼 수 있다.
3. **실패 사례**: 외형 변화가 극심한 포즈 변화나, 매우 유사한 객체들이 밀집해 있는 환경(예: 수족관의 물고기들)에서는 여전히 ID 스위칭이나 분리 실패 문제가 발생한다.

## 📌 TL;DR

본 논문은 비디오 내 객체의 **탐지, 세그멘테이션, 추적을 동시에 수행하는 Video Instance Segmentation(VIS)** 태스크를 최초로 제안하고, 이를 위한 대규모 벤치마크인 **YouTube-VIS**와 알고리즘인 **MaskTrack R-CNN**을 제시하였다. 제안된 방법은 Mask R-CNN에 추적 전용 브랜치와 메모리 큐를 추가하여 우수한 성능을 보였으며, 실험을 통해 성능 향상을 위해서는 단순한 추적 기법 개선보다 이미지 레벨의 정밀한 예측 성능 향상이 더 중요함을 밝혀냈다. 이 연구는 향후 비디오 이해(Video Understanding) 연구에서 객체 단위의 정밀한 분석을 가능케 하는 중요한 이정표가 될 것으로 보인다.
