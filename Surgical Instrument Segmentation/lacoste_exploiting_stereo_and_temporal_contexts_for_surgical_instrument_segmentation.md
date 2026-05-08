# LACOSTE: Exploiting stereo and temporal contexts for surgical instrument segmentation

Qiyuan Wang, Shang Zhao, Zikang Xu, S Kevin Zhou (2024)

## 🧩 Problem to Solve

본 논문은 최소 침습 수술(Minimally Invasive Surgeries)에서 핵심적인 역할을 하는 수술 도구 분할(Surgical Instrument Segmentation, SIS) 문제를 다룬다. 수술 도구 분할은 수술 결정 지원, 내비게이션, 숙련도 평가 및 증강 현실(AR) 환경 구축과 같은 다양한 하위 작업의 필수적인 전제 조건이다.

그러나 수술 장면의 특성상 다음과 같은 기술적 난제가 존재한다. 첫째, 서로 다른 도구 간의 외형 차이가 적은 저간 클래스 분산(low inter-class variance)과 동일 도구라도 자세에 따라 외형이 크게 변하는 고간 클래스 분산(high intra-class variance)이 동시에 나타난다. 둘째, 수술 중 발생하는 연기, 혈액으로 인한 가려짐(occlusion), 조명 변화, 모션 블러(motion blur) 등이 정확한 분할을 방해한다.

기존의 많은 방법론은 수술 비디오의 자연스러운 속성인 시간적(temporal) 흐름과 스테레오(stereo) 정보를 무시하고, 단일 프레임 기반의 인스턴스 분할로 문제를 정의하였다. 이로 인해 시간적 움직임이나 시점 변화에 따른 외형 변화에 취약하며, 특히 도구의 오분류(misclassification) 문제가 빈번하게 발생한다는 한계가 있다. 따라서 본 논문의 목표는 스테레오 및 시간적 컨텍스트를 통합적으로 활용하여 수술 도구 분할의 강건성과 정확도를 높이는 LACOSTE 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Query-Based Segmentation (QBS) 패러다임을 기반으로, 위치 편향(location bias)을 제거하고 스테레오-시간적 컨텍스트를 활용하여 분류 성능을 극대화하는 것이다. 주요 기여 사항은 다음과 같다.

1. **DFP (Disparity-Guided Feature Propagation) 모듈**: 스테레오 쌍의 시차(disparity) 정보를 활용하여 깊이 인식 특징(depth-aware features)을 강화한다. 특히 단안(monocular) 비디오에서도 적용 가능하도록 Pseudo Stereo 생성 메커니즘을 도입하였다.
2. **STSCls (Stereo-Temporal Set Classifier)**: 단일 프레임의 예측이 아닌, 시간 및 스테레오 차원에서 동일한 정체성을 가진 쿼리들의 집합인 '트랙렛(tracklet)'을 구성하고 이를 통해 통합된 분류 결정을 내림으로써 일시적인 예측 실패를 방지한다.
3. **LACls (Location-Agnostic Classifier)**: 마스크 예측과 클래스 분류를 분리하여, 쿼리 임베딩에 포함된 위치 편향을 제거하고 순수하게 세만틱한 특징만을 사용하여 분류 정확도를 높인다.
4. **쿼리 정렬 메커니즘 및 정체성 정렬 손실(Identity Alignment Loss)**: 비디오 수준의 정답(ground truth) ID가 없는 데이터셋에서도 쿼리 인덱스를 통해 정체성을 유지하며 트랙렛을 생성할 수 있는 체계를 구축하였다.

## 📎 Related Works

### 기존 접근 방식 및 한계

- **CNN 기반 SIS**: U-Net 및 Mask R-CNN 기반의 방법들이 제안되었으나, 주로 픽셀 단위의 분류에 집중하여 인스턴스의 특성을 무시하거나 끊어진 마스크를 생성하는 경향이 있었다.
- **Transformer 기반 SIS**: 최근 Vision Transformer(ViT)와 Mask2Former와 같은 QBS 구조가 도입되어 성능이 향상되었다. 하지만 여전히 단일 프레임 기반 예측에 의존하여 시간적 일관성이 부족하다는 점이 지적되었다.
- **시간성 강화 SIS**: Optical Flow나 Video Transformer를 사용하여 시간적 정보를 통합하려는 시도가 있었으나, 전역적인 비디오 컨텍스트를 충분히 활용하지 못하거나 복잡한 계산 비용이 발생하는 문제가 있었다.

### LACOSTE의 차별점

LACOSTE는 단순히 시간적 정보를 추가하는 것에 그치지 않고, 스테레오 시차 정보를 특징 수준에서 융합(DFP)하고, 트랙렛 기반의 집합 분류기(STSCls)를 통해 결정을 공고히 하며, 위치 편향을 제거하는 전용 분류기(LACls)를 계층적으로 배치함으로써 기존 방법론들이 간과한 '위치 편향'과 '스테레오-시간적 통합' 문제를 동시에 해결하였다.

## 🛠️ Methodology

### 전체 시스템 구조

LACOSTE의 추론 파이프라인은 세 단계의 순차적/병렬적 단계로 구성된다.

1. **Frame Step**: DFP가 적용된 QBS 베이스라인($B_{DFP}$)이 각 타임스탬프에서 스테레오 프레임을 입력받아 프레임별 마스크와 초기 클래스 예측을 수행한다.
2. **Tracklet Step**: Frame Step에서 생성된 쿼리들을 시간 및 스테레오 차원에서 묶어 트랙렛을 생성하고, STSCls가 이를 입력받아 통합된 클래스 예측을 수행한다.
3. **L-Agnostic Step**: Frame Step의 마스크 결과에 따라 이미지를 크롭(crop)하고, LACls가 위치 정보가 제거된 특징만을 추출하여 클래스를 예측한다.

최종 결과는 위 세 단계의 예측 확률을 가중 합산(ensemble)하여 결정하며, 마스크는 Frame Step의 결과를 그대로 사용한다.

### 주요 구성 요소 및 상세 설명

#### 1. Disparity-Guided Feature Propagation (DFP)

DFP는 스테레오 우측 영상의 특징을 좌측 영상의 특징으로 투영하여 융합한다. 오프라인 광학 흐름(optical flow) 네트워크 $\Phi$를 통해 시차를 추정하고, Backward Warping 함수 $W$를 사용하여 우측 특징 $F_R$을 좌측 시점으로 정렬한다.
$$F_{R \to L}^{(t)} = W(F_R^{(t)}, \Phi(I_L, I_R))$$
정렬된 특징은 좌측 특징 $F_L$과 픽셀 단위 코사인 유사도 가중치 $w_{R \to L}$를 통해 적응적으로 융합된다.
$$F_{DFP}^{(t)} = F_L^{(t)} + w_{R \to L} F_{R \to L}^{(t)}$$
단안 영상의 경우, 깊이 추정 네트워크를 통해 Pseudo Right 영상을 생성하여 동일한 프로세스를 적용한다.

#### 2. Stereo-Temporal Set Classifier (STSCls)

STSCls는 동일한 정체성을 가진 쿼리 임베딩의 집합인 트랙렛 $s_n$을 입력으로 받는다.

- **쿼리 정렬(Query Alignment)**: 이전 프레임의 쿼리 임베딩을 다음 프레임의 초기 쿼리로 사용하여 정체성을 유지한다.
- **구조**: Transformer 레이어를 쌓아 구성되며, 학습 가능한 set classification token $x_0$가 트랙렛 내의 인스턴스 토큰들로부터 정보를 집계하여 최종 클래스 $p_s$를 예측한다.
- **손실 함수**: 전역 집합 분류 손실 $\mathcal{L}_{sc}$, 지역 인스턴스 분류 손실 $\mathcal{L}_{lc}$, 그리고 동일 정체성 쿼리 간의 유사도를 높이는 정체성 정렬 손실 $\mathcal{L}_{ida}$의 합으로 정의된다.
$$\mathcal{L}_{STSCls} = \mathcal{L}_{sc} + \mathcal{L}_{lc} + \mathcal{L}_{ida}$$

#### 3. Location-Agnostic Classifier (LACls)

QBS의 쿼리는 세만틱 정보뿐만 아니라 위치 편향(location bias)을 함께 학습하므로, 이를 분리하기 위해 제안되었다. BDFP가 예측한 마스크 영역을 기반으로 원본 이미지에서 해당 영역만 크롭하여 사전 학습된 ViT에 입력한다. 이를 통해 배경이나 상대적 위치 정보 없이 오직 도구의 외형적 특징만으로 클래스를 분류하게 함으로써 위치 편향을 제거한다.

### 학습 절차

- $\text{BDFP}$와 $\text{STSCls}$는 $\mathcal{L}_{total} = \mathcal{L}_{baseline} + \mathcal{L}_{STSCls}$ 손실 함수를 통해 공동 학습된다.
- $\text{LACls}$는 학습 비용 절감을 위해 오프라인으로 한 번만 학습시켜 플러그인 형태로 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: EndoVis 2017, EndoVis 2018, GraSP 세 가지 공개 데이터셋을 사용하였다.
- **지표**: Challenge IoU ($\text{ChIoU}$), ISINet IoU ($\text{ISIIoU}$), mean class IoU ($\text{mcIoU}$)를 사용하였다.
- **비교 대상**: S3Net, QPD, MATIS 등 최신 SOTA 모델들과 비교하였다.

### 정량적 결과

- **EndoVis 2017/2018**: LACOSTE(L, Swin-Large backbone) 모델은 모든 지표에서 SOTA를 달성하였다. 특히 EV17에서 QPD 대비 $\text{mcIoU}$가 28% 향상되었으며, 시간적 일관성을 강조한 MATIS보다도 월등한 성능을 보였다. 이는 스테레오-시간적 정보의 융합이 도구 간의 변별력을 크게 높였음을 시사한다.
- **GraSP**: 다양한 수술 도메인에 대한 일반화 성능을 확인한 결과, $\text{mcIoU}$ 80.07, $\text{ISIIoU}$ 84.81을 기록하며 최상위권의 성능을 보였다. 특히 인스턴스 분할 지표인 $\text{AP}_{50}^{segm}$에서 매우 우수한 결과를 나타냈다.

### 주요 분석 결과

- **Ablation Study**: DFP $\to$ STSCls $\to$ LACls 순으로 모듈을 추가할 때마다 성능이 단계적으로 향상됨을 확인하였다. 특히 $\text{mcIoU}$의 경우 STSCls의 기여도가 가장 컸으며, $\text{ChIoU}$는 LACls가 효과적이었다.
- **Ensemble 효과**: Frame, Tracklet, L-Agnostic 세 단계의 예측을 앙상블했을 때 단일 단계만 사용했을 때보다 모든 지표에서 성능이 상승하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문은 t-SNE 분석을 통해 $\text{LACls}$가 어떻게 위치 편향을 제거하는지 시각적으로 증명하였다. BDFP나 STSCls의 쿼리 공간에서는 동일 도구라도 위치나 방향에 따라 클러스터가 분산되는 경향이 있었으나, LACls를 거친 후에는 하나의 조밀한 클러스터로 통합되었다. 이는 위치 정보를 배제하고 순수 세만틱 특징만을 추출하는 전략이 유효했음을 보여준다.

또한, 정체성 정렬 메커니즘을 통해 비디오 수준의 어노테이션 없이도 쿼리 인덱스만으로 일관된 트랙렛을 생성할 수 있음을 입증하였다. 이는 정답 레이블링 비용이 매우 높은 의료 영상 분야에서 매우 실용적인 접근 방식이다.

### 한계 및 비판적 해석

- **추론 시간**: 스테레오 및 시간적 컨텍스트를 처리하기 위해 추가적인 계산량이 발생한다. 비록 메모리 뱅크를 통해 최적화하였으나, 실시간 수술 보조 시스템에 적용하기 위해서는 TensorRT와 같은 추가적인 가속화 최적화가 필수적일 것으로 보인다.
- **Pseudo Stereo의 한계**: 단안 영상에서 생성한 가상 스테레오 영상은 실제 스테레오 영상보다 성능이 약간 낮게 나타났다. 이는 가상 영상의 엣지 부분의 불완전함과 노이즈 때문으로 분석되며, 더 정교한 깊이 추정 모델이 필요할 수 있다.
- **VOS와의 관계**: SAM2와 같은 Video Object Segmentation(VOS) 모델과 비교했을 때, LACOSTE는 참조 프레임의 정답 없이도 작동한다는 장점이 있지만, VOS 특유의 정밀한 마스크 전파 능력과는 지향점이 다르다는 점을 명시하고 있다.

## 📌 TL;DR

LACOSTE는 수술 도구 분할에서 발생하는 **오분류 문제**를 해결하기 위해 **스테레오 시차(DFP)**, **시간적 트랙렛(STSCls)**, 그리고 **위치 편향 제거(LACls)**라는 세 가지 전략을 결합한 프레임워크이다. 쿼리 기반 분할 방식을 확장하여 비디오의 시간적/공간적 문맥을 통합적으로 활용함으로써, 기존 SOTA 모델들을 뛰어넘는 높은 분류 정확도와 분할 성능을 달성하였다. 이 연구는 특히 정답 ID가 없는 수술 비디오에서도 강건한 인스턴스 추적이 가능함을 보여주었으며, 향후 실시간 수술 내비게이션 및 자동화 시스템의 핵심 모듈로 활용될 가능성이 높다.
