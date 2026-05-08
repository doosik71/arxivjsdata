# PerSense: Training-Free Personalized Instance Segmentation in Dense Images

Muhammad Ibraheem Siddiqui et al. (2025)

## 🧩 Problem to Solve

본 논문은 객체들이 매우 조밀하게 배치된 **Dense Images(밀집 이미지)** 환경에서 특정 시각적 범주나 개념을 분리해내는 **Personalized Instance Segmentation(개인화된 인스턴스 분할)** 문제를 해결하고자 한다. 밀집 이미지 환경에서는 객체 간의 심한 가려짐(Occlusion), 규모의 다양성(Scale variations), 그리고 배경 clutter로 인해 정밀한 인스턴스 경계 획정(Delineation)이 매우 어렵다.

기존의 Foundation Model인 SAM(Segment Anything Model)의 'Everything mode'는 배경과 전경을 구분 없이 모두 분할하며, 사용자가 수동으로 프롬프트를 제공해야 하는 번거로움이 있다. 또한 Grounded-SAM과 같이 Bounding Box 프롬프트를 사용하는 방식은 박스 형태의 제약과 NMS(Non-Maximum Suppression) 과정에서의 한계로 인해, 밀집된 환경에서 인접한 여러 인스턴스를 하나의 박스로 묶어버리는 문제가 발생한다. 따라서 본 논문의 목표는 추가 학습이 필요 없는(Training-free), 모델 불가지론적(Model-agnostic)인 **One-shot 프레임워크**를 통해 밀집 이미지 내의 개인화된 인스턴스를 자동으로 정밀하게 분할하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Density Map(DM, 밀도 맵)**을 활용하여 인스턴스 수준의 정밀한 **Point Prompt(점 프롬프트)**를 자동으로 생성하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Instance Detection Module (IDM)**: 밀도 맵을 통해 객체의 공간적 분포를 파악하고, 이를 기반으로 개별 인스턴스를 대표하는 후보 점 프롬프트를 생성한다.
2. **Point Prompt Selection Module (PPSM)**: 객체의 밀도에 따라 동적으로 변하는 **Adaptive Threshold(적응형 임계값)**와 **Box-gating** 메커니즘을 도입하여 IDM에서 생성된 후보 점 중 False Positive(오탐)를 효과적으로 제거한다.
3. **Feedback Mechanism**: 초기 분할 결과를 바탕으로 최적의 Exemplar(예시)를 다시 선택하여 밀도 맵의 품질을 개선함으로써 최종 분할 정확도를 높인다.
4. **PerSense-D 벤치마크**: 밀집 이미지 환경에 특화된 새로운 One-shot 분할 벤치마크를 제안하여, 기존 데이터셋(COCO, LVIS 등)이 다루지 못한 고밀도 시나리오의 평가 기준을 마련하였다.

## 📎 Related Works

### 기존 연구 및 한계

- **One-shot Personalized Segmentation**: PerSAM은 반복적인 마스킹을 통해 학습 없이 분할을 수행하지만, 밀집 환경에서는 계산 비용이 급증하고 마스킹된 객체가 많아질수록 신뢰도 맵(Confidence map)의 품질이 저하되는 문제가 있다. SegGPT나 Painter 같은 모델은 In-context prompting을 사용하지만, 밀집된 영역에서 객체 간의 분리를 명확히 하지 못하고 단순화하는 경향이 있다. Matcher는 특성 매칭과 클러스터링을 사용하지만, 여전히 Box 프롬프트의 한계로 인해 밀집 씬에서의 성능이 제한적이다.
- **Interactive Segmentation**: InterFormer, SEEM 등은 높은 정밀도를 제공하지만, 수동 프롬프트에 의존하므로 확장성이 부족하며 단일 참조 이미지로부터 모든 인스턴스를 일반화하여 분할하는 기능이 부족하다.

### PerSense의 차별점

PerSense는 클러스터링이나 수동 입력 대신, **DMG(Density Map Generator)**를 통해 개인화된 밀도 맵을 생성하고 여기서 직접 점 프롬프트를 추출한다. 이는 밀집 환경에서 객체 간의 분리 성능을 비약적으로 향상시키며, 단일 패스(Single pass)만으로 효율적인 처리가 가능하다는 점에서 기존의 반복적 방식(PerSAM 등)과 차별화된다.

## 🛠️ Methodology

PerSense는 전체적으로 학습이 필요 없는 파이프라인으로 구성되며, 주요 단계는 다음과 같다.

### 1. Class-label Extraction 및 Exemplar Selection

먼저 Support 이미지에서 객체 범주를 추출하기 위해 CLE(Class-label Extractor)를 사용하며, "Name the object in the image?"라는 프롬프트를 통해 명사 형태의 클래스 라벨을 얻는다. 이후 Grounding Detector를 통해 Query 이미지 내의 객체들을 탐지하고, Support 특징과 Query 특징 간의 코사인 유사도(Cosine Similarity)를 계산하여 가장 높은 유사도를 가진 지점을 초기 긍정 위치(Positive location prior) $P_{max}$로 설정한다.

$$S_{score}(Q, S_{supp}) = \text{cos\_sim}(f(Q), f(S_{supp})), \quad P_{max} = \arg \max_{P \in B_{max}} S_{score}(P, S_{supp})$$

여기서 $f(\cdot)$는 인코더를 의미하며, $B_{max}$는 가장 신뢰도가 높은 바운딩 박스이다.

### 2. Instance Detection Module (IDM)

IDM은 밀도 맵(DM)을 입력받아 점 프롬프트를 생성한다. 과정은 다음과 같다.

1. **이진화 및 침식**: DM을 그레이스케일 이미지 $I_{gray}$로 변환 후 임계값 $T$를 적용해 이진 이미지 $I_{binary}$를 만들고, $3 \times 3$ 커널을 이용해 침식(Erosion) 연산을 수행하여 노이즈를 제거한다.
2. **Composite Contour 처리**: 추출된 컨투어(Contour)의 면적 $A_{ctr}$이 가우시안 분포를 따른다고 가정하고, $\mu + 2\sigma$를 초과하는 면적을 가진 컨투어를 '복합 컨투어(Composite contour)'로 정의한다.
3. **Distance Transform**: 복합 컨투어 내부의 개별 객체를 분리하기 위해 거리 변환(Distance Transform)을 적용하고, 다시 이진 임계값을 통해 내부의 하위 영역(Sub-regions)을 분리한다.
4. **Centroid 추출**: 각 컨투어의 공간 모멘트(Spatial moments)를 계산하여 중심점(Centroid)을 구하며, 이 점들이 최종 후보 점 프롬프트가 된다.

$$\text{Centroid: } c_X = \frac{M_{10}}{M_{00} + \epsilon}, \quad c_Y = \frac{M_{01}}{M_{00} + \epsilon}$$

### 3. Point Prompt Selection Module (PPSM)

IDM에서 생성된 후보 점 중 오탐을 제거하기 위해 적응형 임계값 $T_{adapt}$를 사용한다.

$$T_{adapt} = \frac{S_{max}}{C/k}, \quad \text{for } C > 1$$

여기서 $S_{max}$는 최대 유사도 점수, $C$는 객체 수, $k$는 정규화 상수($\sqrt{2}$)이다. 객체 수가 많아질수록 임계값이 낮아져(더 관대해져) 밀집 환경에서 True Positive를 놓치지 않도록 설계되었다. 최종적으로 점은 **(1) 유사도 점수가 $T_{adapt}$보다 높고, (2) Grounding Detector가 예측한 박스 내에 위치**해야 선택된다.

### 4. Feedback Mechanism

초기 분할 결과($M_{seg}$)와 SAM이 생성한 마스크 점수($S_{mask}$)를 기반으로, 상위 $m=4$개의 최적 마스크를 선택하여 이를 다시 DMG의 Exemplar로 사용한다. 이를 통해 더 정교한 밀도 맵을 재생성하고, 결과적으로 더 정확한 점 프롬프트를 추출하는 선순환 구조를 구축한다.

## 📊 Results

### 실험 설정

- **데이터셋**: PerSense-D (신규), COCO-20i, LVIS-92i.
- **지표**: mIoU (mean Intersection over Union).
- **구성**: VIP-LLaVA(CLE), GroundingDINO(Detector), SAM(Segmentation), DSALVANet/CounTR(DMG).

### 주요 결과

1. **밀집 벤치마크(PerSense-D)**: PerSense는 전반적인 클래스별 mIoU **71.61%**를 기록하며 PerSAM(+47.16%), SegGPT(+16.11%), Matcher(+8.83%) 등 SOTA 모델을 크게 앞질렀다. 특히 객체 밀도가 높아질수록(Low $\rightarrow$ Medium $\rightarrow$ High) 성능이 단조 증가하는 경향을 보이며 밀집 씬에서의 강점을 입증했다.
2. **희소 벤치마크(COCO, LVIS)**: 밀집 환경에 특화되었음에도 불구하고 COCO와 LVIS에서도 경쟁력 있는 성능을 보였다. 다만, 데이터셋 특성상 전통적인 탐지기 기반 방법론(Matcher 등)이 상대적으로 우세한 경향이 있었다.
3. **효율성**: 추론 시간은 약 **2.7초**로 Matcher나 PerSAM보다 빠르며, Grounded-SAM과 유사한 수준의 오버헤드만 발생시킨다.
4. **Ablation Study**: PPSM은 오탐을 줄여 mIoU를 약 1.37~2.48% 향상시켰으며, Feedback Mechanism은 밀집 환경에서 특히 큰 효과(+4.01%)를 보였다.

## 🧠 Insights & Discussion

### 강점

- **학습 프리(Training-free)** 및 **모델 불가지론(Model-agnostic)** 설계로 인해, 특정 도메인에 국한되지 않고 의료 영상(Cellular segmentation) 등 다양한 분야로의 확장이 용이하다.
- 밀도 맵이라는 전역적 정보와 점 프롬프트라는 지역적 정보를 결합하여, 기존 Box 기반 방식이 해결하지 못한 밀집 환경의 오버랩 문제를 효과적으로 해결했다.

### 한계 및 비판적 해석

- **DMG 의존성**: 본 프레임워크의 가장 큰 취약점은 밀도 맵 생성기(DMG)에 의존한다는 점이다. 실험 결과(Failure cases)에서 나타나듯, DMG 단계에서 객체를 놓치면(False Negative), 이후의 PPSM이나 Feedback 과정에서도 이를 복구할 방법이 없다. 즉, 시스템의 성능 상한선이 사용된 DMG의 성능에 묶여 있다.
- **Feedback 반복 횟수**: 논문에서는 단일 패스의 Feedback만으로 충분하다고 주장하지만, 이는 SAM의 마스크 품질이 매우 높다는 가정 하에 성립한다.

## 📌 TL;DR

PerSense는 밀집된 이미지 내의 특정 객체를 분할하기 위해 **밀도 맵(Density Map)을 활용하여 자동으로 정밀한 점 프롬프트를 생성**하는 학습 프리(Training-free) One-shot 프레임워크이다. IDM과 PPSM을 통해 밀집 환경의 난제인 객체 중첩 및 오탐 문제를 해결했으며, 신규 벤치마크인 PerSense-D를 통해 그 우수성을 입증했다. 이 연구는 산업용 품질 관리나 의료 세포 분할과 같이 객체가 매우 조밀하게 배치된 실무 환경의 자동화에 크게 기여할 가능성이 높다.
