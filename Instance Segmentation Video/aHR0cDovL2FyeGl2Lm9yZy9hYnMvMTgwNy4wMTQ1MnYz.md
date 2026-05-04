# Semantic Instance Meets Salient Object: Study on Video Semantic Salient Instance Segmentation

Trung-Nghia Le, Akihiro Sugimoto (2018)

## 🧩 Problem to Solve

본 논문은 비디오 내에서 단순히 '두드러진 영역'을 찾는 것을 넘어, 그 영역을 의미론적으로 유의미한 개별 구성 요소로 분리하는 **Video Semantic Salient Instance Segmentation (VSSIS)**라는 새로운 과제를 정의하고 해결하고자 한다.

기존의 비디오 Salient Object Segmentation (SOS)은 픽셀 단위로 'salient' 또는 'non-salient'로만 라벨링하여 관심 영역을 국지화하는 데 집중했다. 하지만 실제 환경에서는 하나의 두드러진 영역 내에 여러 개의 상호작용하는 객체가 포함될 수 있으며, 이를 개별적인 인스턴스로 분리하고 각각에 semantic label을 부여하는 것이 더 고도화된 비디오 이해를 가능하게 한다.

특히 자율주행 자동차나 로봇 내비게이션의 경우, 장면 전체의 모든 객체를 분석하는 것보다 도로 위의 보행자나 차량과 같이 유용하고 두드러진(salient) semantic instance에만 집중하는 것이 연산 효율성과 정확도 측면에서 훨씬 유리하다. 따라서 본 연구의 목표는 비디오에서 두드러진 전경 객체 클래스만을 개별 인스턴스로 식별하고 세그멘테이션하는 VSSIS의 기준 모델(baseline)을 제시하고, 이를 위한 데이터셋을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같이 세 가지로 요약할 수 있다.

1. **VSSIS 과제 정의**: 기존의 Semantic Instance Segmentation (SIS)과 Salient Object Segmentation (SOS)을 결합하여, 비디오 내에서 의미론적으로 중요한 두드러진 인스턴스만을 추출하는 VSSIS라는 새로운 태스크를 제안하고 분석하였다.
2. **SISO 프레임워크 제안**: SIS와 SOS 두 스트림의 장점을 결합한 **Semantic Instance - Salient Object (SISO)** 베이스라인을 제안하였다. 이 프레임워크는 Sequential Fusion, Recurrent Instance Propagation, Identity Tracking이라는 세 가지 핵심 메커니즘을 통해 인스턴스의 일관성을 유지하고 정밀도를 높인다.
3. **SESIV 데이터셋 구축**: VSSIS 연구를 위해 DAVIS-2017 벤치마크를 확장하여, 84개의 고품질 비디오 시퀀스로 구성된 **SEmantic Salient Instance Video (SESIV)** 데이터셋을 구축하고 공개하였다. 이 데이터셋은 픽셀 수준의 saliency, instance, semantic 라벨을 모두 포함한다.

## 📎 Related Works

본 논문은 크게 두 가지 관련 연구 흐름을 언급한다.

- **Semantic Instance Segmentation (SIS)**: 객체 탐지와 semantic segmentation을 통합하는 작업으로, Mask R-CNN과 같은 proposal 기반 접근 방식이 주류를 이룬다. 그러나 기존 연구들은 이미지 단위에 집중되어 있으며, 비디오 전체에서 인스턴스를 일관되게 추적하는 VSSIS 관점의 연구는 부재한 상태이다.
- **Video Salient Object Segmentation (VSOS)**: CNN 기반의 FCN(Fully Convolutional Networks)이나 3D 커널, Optical Flow를 활용하여 비디오 내의 두드러진 영역을 찾는 연구들이 진행되었다. 하지만 이러한 방법들은 영역(region)을 찾는 데 그칠 뿐, 내부의 개별 인스턴스를 분리하거나 semantic label을 부여하는 단계까지는 나아가지 못했다.

SISO는 이러한 두 분야의 간극을 메우기 위해, SOS를 통해 '어디가 중요한가'를 파악하고 SIS를 통해 '그곳에 무엇이 있는가'를 정의함으로써 두 태스크를 통합적으로 수행한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

SISO는 SIS 스트림과 SOS 스트림으로 구성된 2-스트림 프레임워크이다. 전체 파이프라인은 각 스트림의 출력을 융합하고, 시간적 도메인에서 인스턴스를 전파 및 추적하는 구조로 설계되었다.

### 1. 전체 구조 및 스트림 구성

- **SIS Stream**: 프레임별로 semantic instance를 세그멘테이션한 후, Recurrent Instance Propagation을 통해 비디오 전체로 전파하여 형태의 정확도를 높인다.
- **SOS Stream**: 3D FCN 모델(DSRFCN3D)을 사용하여 saliency map을 생성하며, 이를 적응형 임계값 $\theta = \mu + \eta$ (평균 $\mu$, 표준편차 $\eta$)를 통해 이진화하여 salient region mask를 획득한다.

### 2. Sequential Fusion (순차적 융합)

SIS 스트림에서 검출된 여러 인스턴스가 서로 겹칠 때 발생하는 문제를 해결하기 위해, SOS 스트림의 mask를 기준으로 융합 순서를 결정한다.

- **절차**: Salient region mask $M$과 가장 높은 $IOU$를 가진 인스턴스를 먼저 선택하여 fusion map에 배치하고, 해당 영역을 mask $M$에서 제거한다. 이 과정을 mask 내에 인스턴스가 더 이상 존재하지 않을 때까지 반복한다.
- **신뢰도 계산**: 각 프레임의 신뢰도(frame-confidence, $FC$)는 선택된 인스턴스들의 confident score $CS$의 평균으로 계산하며, $CS$는 다음과 같이 segmentation $IOU$ score ($S^{(seg)}$)와 classification accuracy score ($S^{(cls)}$)의 조화 평균으로 정의된다.
$$CS = \frac{(1 + \beta^2) S^{(seg)} S^{(cls)}}{\beta^2 S^{(seg)} + S^{(cls)}}$$
(실험적으로 $\beta^2 = 0.3$을 사용하여 segmentation 점수에 더 높은 가중치를 부여하였다.)

### 3. Recurrent Instance Propagation (재귀적 인스턴스 전파)

객체의 변형이나 카메라 움직임으로 인해 일부 프레임에서 인스턴스가 누락되는 문제를 해결한다.

- **절차**: 프레임들을 $FC$ 값이 높은 순서대로 정렬한 후, 신뢰도가 높은 프레임의 인스턴스를 FlowNet2 기반의 flow warping/inverse flow warping을 통해 인접한 신뢰도가 낮은 프레임으로 전파한다.
- **반복**: 이 과정은 비디오 전체의 평균 신뢰도가 수렴할 때까지 재귀적으로 수행되며, 일반적으로 약 5회 반복 후 효과적으로 전파된다.

### 4. Identity Tracking (아이덴티티 추적)

비디오 전체에서 인스턴스의 ID와 semantic label의 일관성을 유지한다.

- **Identity Propagation**: 첫 프레임에서 ID를 초기화하고, flow warping을 통해 다음 프레임으로 전달한다. $IOU$가 $0.7$ 미만인 경우 객체가 가려졌거나(occluded) 화면 밖으로 나간 것으로 간주한다.
- **Re-identification**: 누락된 인스턴스를 찾기 위해 key-frame(평균 면적이 최대인 프레임)의 특징 벡터를 추출하고, Faster R-CNN의 region proposal들과 코사인 유사도를 비교하여 동일 객체를 재식별한다.
- **Semantic Unification**: 비디오 전체에 걸쳐 해당 인스턴스가 가졌던 classification score의 합이 가장 높은 카테고리를 최종 semantic label로 결정하여 라벨을 통일한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 직접 구축한 SESIV 데이터셋 (학습 58개, 테스트 26개 비디오).
- **모델**: SOS 스트림은 DSRFCN3D를, SIS 스트림은 Mask R-CNN과 MNC를 사용하여 베이스라인 성능을 측정하였다.
- **평가 지표**: Semantic Region Similarity ($J_S$)와 Semantic Contour Accuracy ($F_S$)를 도입하였다. 두 지표 모두 ID와 semantic label이 일치하는 경우에만 기존의 $J$ (region similarity)와 $F$ (contour accuracy)를 계산하여 평균을 낸다.
$$J_S(m,g) = \delta_{id(m),id(g)} \delta_{sl(m),sl(g)} J(m,g)$$
$$F_S(m,g) = \delta_{id(m),id(g)} \delta_{sl(m),sl(g)} F(m,g)$$

### 2. 주요 결과

- **SISO의 효과**: Table 2에 따르면, Mask R-CNN이나 MNC 단독 사용(frame-by-frame) 또는 단순 전파($prop$) 방식보다 SISO 프레임워크를 적용했을 때 $J_S$와 $F_S$ 지표가 비약적으로 상승하였다. 이는 SISO가 non-salient 인스턴스를 효과적으로 제거하고 ID 일관성을 잘 유지함을 보여준다.
- **Ablation Study**:
  - **Confident-Instance 활용**: 무작위 순서로 융합하는 것보다 제안한 sequential fusion과 recurrent propagation을 모두 사용했을 때 성능이 가장 높았다 (Table 3).
  - **Identity Tracking**: 단순 전파보다 Re-identification을 포함한 전체 추적 모듈을 사용했을 때 성능 향상이 뚜렷했으며, 특히 추적을 하지 않았을 때($SISO_\alpha$)와 비교해 매우 큰 성능 차이를 보였다 (Table 4).

## 🧠 Insights & Discussion

본 논문은 VSSIS라는 새로운 태스크를 정의하고, 기존의 SOS와 SIS 모델을 효율적으로 결합하는 파이프라인을 구축함으로써 가능성을 입증하였다. 특히, 단순한 프레임별 처리의 한계를 극복하기 위해 **'신뢰도 기반의 재귀적 전파'**와 **'특징 기반의 재식별(Re-ID)'** 메커니즘을 도입한 점이 고무적이다.

하지만 본 연구는 기존의 pre-trained 모델들을 그대로 사용한 baseline 연구라는 한계가 있다. 즉, VSSIS 태스크에 최적화되어 end-to-end로 학습된 모델이 아니라, 기존 모델들의 출력을 후처리 단계에서 융합하고 전파하는 방식이다. 또한 SESIV 데이터셋의 규모가 84개 비디오로 제한적이어서, 더 방대한 데이터셋에서의 일반화 성능 검증이 필요하다.

결론적으로 본 논문은 "중요한 객체만 골라내어 세그멘테이션한다"는 실용적인 관점을 제시하였으며, 이는 향후 자율주행 및 로보틱스 분야에서 연산 비용을 줄이면서도 핵심 정보를 추출하는 연구의 기초가 될 수 있다.

## 📌 TL;DR

본 논문은 비디오 내에서 의미론적으로 중요한 두드러진 객체만을 분리하는 **Video Semantic Salient Instance Segmentation (VSSIS)** 태스크를 제안하고, 이를 해결하기 위한 **SISO** 프레임워크와 **SESIV** 데이터셋을 구축하였다. SISO는 SOS와 SIS의 출력을 순차적으로 융합하고, 재귀적 전파와 ID 추적을 통해 가려짐(occlusion) 등의 문제를 해결하며 높은 일관성을 확보하였다. 이 연구는 자율주행 및 로봇 시스템이 장면 전체가 아닌 핵심 객체에만 집중하게 함으로써 효율성을 높이는 데 기여할 가능성이 크다.
