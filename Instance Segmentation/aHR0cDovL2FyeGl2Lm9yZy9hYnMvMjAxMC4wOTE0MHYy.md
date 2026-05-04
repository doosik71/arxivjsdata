# Localized Interactive Instance Segmentation

Soumajit Majumder, Angela Yao (2020)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 기존의 대화형 인스턴스 분할(Interactive Instance Segmentation) 방식에서 사용자 인터랙션의 효율성이 떨어진다는 점이다. 기존 방식들은 사용자가 대상 객체와 멀리 떨어진 배경이나 다른 객체에 자유롭게 클릭을 제공할 수 있도록 허용하는데, 이는 대상 객체를 효율적으로 분리하려는 최종 목표와 일치하지 않는 비효율적인 상호작용 방식이다.

또한, 기존의 가이드 맵(Guidance maps) 생성 방식은 Euclidean 거리나 Gaussian 분포와 같이 이미지의 내부 구조(에지, 텍스처 등)를 고려하지 않는 Image-agnostic한 방식에 의존하고 있다. 이러한 한계는 네트워크가 객체의 정확한 위치와 경계를 파악하는 데 더 많은 사용자 클릭을 요구하게 만든다. 따라서 본 논문의 목표는 사용자 클릭 범위를 객체 주변으로 제한하는 새로운 클릭 스킴(Clicking scheme)을 제안하고, 이미지 구조와 일치하는 약한 위치 사전 정보(Weak localization prior)를 생성하여 인터랙션 횟수를 줄이면서도 분할 정확도를 높이는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 설계 아이디어는 사용자의 시선을 객체와 그 주변부로 집중시키고, 이를 통해 네트워크에 강력한 위치 힌트를 제공하는 것이다. 주요 기여 사항은 다음과 같다.

1.  **Localized Clicking Scheme**: 사용자의 클릭 범위를 객체 근처로 제한하여 인터랙션의 효율성을 높이고 네트워크가 집중해야 할 영역을 명확히 한다.
2.  **Superpixel-box Guidance Map**: 사용자의 초기 클릭을 변환하여 객체의 대략적인 위치를 나타내는 약한 위치 사전 정보를 생성한다. 이는 단순한 박스 형태가 아니라 Superpixel 단위를 활용하여 이미지의 저수준 구조(Low-level structures)와 일치하도록 설계되었으며, 추가 클릭이 발생할 때마다 점진적으로 정교해진다.
3.  **성능 향상**: MS COCO를 포함한 여러 벤치마크 데이터셋에서 기존 SOTA(State-of-the-art) 모델 대비 더 적은 클릭 횟수로 목표 mIoU에 도달함을 입증하였다.

## 📎 Related Works

기존의 대화형 분할 연구는 크게 세 단계로 구분된다. 초기에는 Active contours(Snakes), Intelligent scissors와 같이 경계 특성에 집중한 방법들이 있었으나, 에지 정보가 약한 경우 성능이 급격히 떨어진다는 한계가 있었다. 이후 Graph cuts나 Geodesics 기반 방법들이 등장했으나, 색상과 텍스처 같은 저수준 특징에만 의존하여 배경과 객체의 외형이 유사한 경우 취약했다.

최근에는 CNN 기반의 딥러닝 아키텍처가 도입되었다. iFCN 등의 연구는 사용자 클릭을 Euclidean 거리 맵으로 변환하여 입력으로 사용했으며, 이후 RIS-Net이나 ITIS 등은 반복적인 학습 절차나 더 깊은 네트워크 구조를 통해 성능을 개선했다. 일부 연구에서는 객체의 Bounding box를 직접 크롭(Crop)하여 사용함으로써 성능을 높이려 했으나, 이는 사용자에게 정확한 극점(Extreme points) 클릭을 요구하여 상호작용 속도를 늦추거나, 미리 학습된 객체 검출기(Object detector)가 필요하다는 제약이 있었다. 본 논문은 이러한 '강한 제약(Hard constraint)' 대신 Superpixel을 이용한 '약한 사전 정보(Weak prior)'를 제공함으로써 클래스에 구애받지 않고(Class-agnostic) 유연하게 대응하는 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조
본 시스템은 RGB 이미지와 사용자 클릭으로 생성된 가이드 맵을 입력으로 받아 객체의 마스크를 예측하는 구조이다. 기본 네트워크로는 DeepLab-v2를 사용하며, ResNet-101 백본과 PSPNet(Pyramid Scene Parsing Network) 예측 헤드로 구성된다.

### 주요 구성 요소 및 절차

#### 1. Interaction Loop (상호작용 루프)
사용자의 상호작용을 다음과 같이 제한한다.
- **초기 단계**: 사용자는 객체의 중심에 positive 클릭($c^+_0$)을, 그 주변 배경에 negative 클릭($c^-_0$)을 한 번씩 수행한다. 이 두 클릭을 통해 객체의 대략적인 위치를 나타내는 Enclosing box(포함 박스)가 생성된다.
- **수정 단계**: 이후의 클릭은 위치가 제한된다. Negative 클릭은 추정된 박스 내부에서만, Positive 클릭은 박스 외부에서만 이루어져야 한다. 새로운 Positive 클릭이 발생하면 위치 사전 정보(Bounding box boundary)가 업데이트된다.

#### 2. Superpixel-based Guidance Maps
이미지의 픽셀들을 유사한 그룹으로 묶은 Superpixel(SLIC 알고리즘 사용)을 활용한다. 클릭된 픽셀이 속한 Superpixel 전체로 가이드 값을 전파하며, 각 픽셀 $p$에 대한 가이드 값 $S^+_Z(p)$는 다음과 같이 계산된다.

$$S^+_Z(p) = \min_{z \in \{z^+\}} d^2_c(z, f^p_Z(p))$$

여기서 $f^p_Z(p)$는 픽셀 $p$가 속한 Superpixel을 반환하는 함수이며, $d^2_c$는 두 Superpixel 중심(Centroid) 간의 Euclidean 거리이다.

#### 3. Superpixel-box Guidance Map (SPBox)
초기 클릭 쌍 $\{c^+_0, c^-_0\}$을 통해 생성된 좌표 범위 $e_0$ (top-left)와 $e_1$ (bottom-right)를 기반으로, 이 범위 내에 포함되는 Superpixel 집합 $\{Z_b\}$를 정의한다. 가이드 맵 $G(p)$는 다음과 같은 지시 함수(Indicator function)로 정의된다.

$$G(p) = 1[p \subset z] \cdot 1[z \subset \{Z_b\}]$$

추가 클릭 $\{c^+_{i \neq 0}, c^-_{i \neq 0}\}$이 들어오면, 새로운 Superpixel 집합 $\hat{Z}_b$를 다음과 같이 업데이트하여 가이드 맵을 정교화한다.

$$\hat{Z}_b = \{Z_b\} \cup \{z^+_{i \neq 0}\} \setminus \{z^-_{i \neq 0}\}$$
$$\hat{G}(p) = 1[p \subset z] \cdot 1[z \subset \hat{Z}_b]$$

#### 4. 학습 절차 및 손실 함수
- **손실 함수**: 예측 마스크와 Ground Truth 마스크 사이의 픽셀 단위 Binary Cross-Entropy (BCE) 손실을 최소화한다.
- **Guidance Dropout**: 네트워크가 클릭 정보에만 과도하게 의존하지 않도록, 학습 중 10%의 확률로 가이드 맵의 값을 255로 고정하거나, 단일 클릭/초기 클릭 쌍만 제공하는 Dropout 기법을 적용한다.

## 📊 Results

### 실험 설정
- **데이터셋**: PASCAL VOC 2012, GrabCut, Berkeley, MS COCO, DAVIS-2016.
- **평가 지표**: 특정 mIoU 임계값(GrabCut/Berkeley는 90%, VOC12/MS COCO는 85%)에 도달하기까지 필요한 평균 클릭 횟수.
- **기준선**: iFCN, RIS-Net, ITIS, DEXTR, VOS-Wild, FCTSFN, IIS-LD, MLG 등.

### 정량적 결과
- **이미지 분할**: GrabCut 데이터셋에서 3.46회로 가장 적은 클릭 횟수를 기록했으며, MS COCO의 Seen category(5.15회)와 Unseen category(5.70회) 모두에서 SOTA 성능을 달성했다. PASCAL VOC 2012에서도 기존 방식 대비 약 7.5%의 성능 향상을 보였다.
- **비디오 분할 (VOS)**: OSVOS의 잘못된 예측 마스크를 수정하는 실험에서, 단 1회의 클릭만으로 mIoU를 50.4%에서 72.2%로 대폭 상승시켜 다른 모든 비교 모델을 압도했다.

### 정성적 분석
Ablation study 결과, Euclidean 거리 맵보다 Superpixel 기반 가이드 맵이 우수하며, 특히 SPBox(Superpixel-box) 가이드를 추가했을 때 클릭 횟수가 가장 크게 감소함을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 사용자의 인터랙션 범위를 제한하는 단순한 전략과 이미지 구조를 반영한 Superpixel-box 가이드 맵의 조합이 인터랙션 효율성을 극대화할 수 있음을 보여주었다. 특히 하드 크롭(Hard crop) 방식과 달리 이미지 전체 맥락을 유지하면서도 객체의 위치 정보를 효과적으로 전달했다는 점이 강점이다.

다만, 논문에서 언급된 한계점은 **객체 간의 폐색(Occlusion)** 상황이다. 두 객체가 겹쳐 있을 경우 Superpixel-box 가이드 영역이 서로 크게 중첩되어, 네트워크가 두 인스턴스를 개별적으로 분리해내는 데 어려움을 겪는 모습이 관찰되었다. 이는 향후 다중 객체 간의 경계 구분 능력을 높이는 연구가 필요함을 시사한다.

## 📌 TL;DR

이 논문은 사용자 클릭 범위를 객체 주변으로 제한하는 **Localized Clicking Scheme**과 Superpixel을 이용해 객체 위치를 알려주는 **Superpixel-box Guidance Map**을 제안하여, 대화형 인스턴스 분할에서 필요한 클릭 횟수를 획기적으로 줄였다. 특히 MS COCO와 같은 어려운 데이터셋에서 SOTA를 달성했으며, 비디오 객체 분할 마스크 수정 작업에서도 탁월한 효율성을 입증하여 향후 정밀한 이미지 어노테이션 도구 및 편집 시스템에 적용될 가능성이 높다.