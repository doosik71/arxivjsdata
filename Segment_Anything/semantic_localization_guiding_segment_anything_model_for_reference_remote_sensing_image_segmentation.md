# Semantic Localization Guiding Segment Anything Model For Reference Remote Sensing Image Segmentation

Shuyang Li, Shuang Wang, Tao Xie, and Zhuangzhuang Sun (2025)

## 🧩 Problem to Solve

본 논문은 참조 원격 탐사 이미지 분할(Reference Remote Sensing Image Segmentation, RRSIS) 작업에서 발생하는 한계점을 해결하고자 한다. RRSIS는 텍스트 설명에 기반하여 이미지 내의 특정 객체에 대한 분할 마스크(segmentation mask)를 생성하는 작업이다.

기존의 RRSIS 방법론들은 주로 멀티모달 퓨전 백본(multi-modal fusion backbones)과 세만틱 분할 헤드(semantic segmentation heads)에 의존해 왔다. 그러나 이러한 접근 방식은 다음과 같은 문제점을 가지고 있다. 첫째, 원격 탐사 이미지의 특성상 자연 이미지에 비해 타겟 객체의 크기가 매우 작고 배경 영역이 매우 복잡하여 정밀한 해석이 어렵다. 둘째, 고품질의 픽셀 수준 주석(pixel-level annotation) 데이터가 부족하여 모델을 충분히 학습시키기에 어려움이 있으며, 이는 학습 데이터에 대한 높은 의존도로 이어진다.

따라서 본 연구의 목표는 텍스트 설명을 통한 객체 위치 추정과 정밀한 마스크 생성을 분리하여, 데이터 주석 부담을 줄이면서도 복잡한 배경 속에서 작은 객체를 정확하게 분할할 수 있는 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 RRSIS 작업을 '거친 위치 추정(coarse localization)'과 '정밀 분할(fine segmentation)'의 두 단계로 분해하는 것이다.

1. **PSLG-SAM 프레임워크 제안**: Visual Grounding 모델을 통해 텍스트 힌트를 좌표 프롬프트로 변환하고, 이를 Segment Anything Model(SAM)의 입력으로 사용하여 정밀한 마스크를 생성하는 2단계 구조를 설계하였다.
2. **CFPG(Clustering-based Foreground Point Generator) 설계**: Visual Grounding 모델이 생성한 바운딩 박스가 부정확할 수 있다는 점을 보완하기 위해, 비지도 학습 기반의 클러스터링을 통해 객체의 중심 전경 포인트(center foreground point)를 생성하여 SAM의 프롬프트로 제공한다.
3. **MBO(Mask Boundary Iterative Optimization) 전략**: GrabCut 알고리즘에서 영감을 얻어, GMM(Gaussian Mixture Model)과 Max-Flow Min-Cut 알고리즘을 이용해 마스크의 경계선을 반복적으로 최적화함으로써 엣지 분할의 정밀도를 높였다.
4. **RRSIS-M 데이터셋 구축**: 전문가의 수동 주석을 통해 마스크 정밀도가 높은 고품질의 멀티 카테고리 원격 탐사 참조 분할 데이터셋을 구축하여 벤치마크로 제공한다.

## 📎 Related Works

### Referring Image Segmentation (RIS)

자연 이미지 분야에서는 이미 많은 발전이 있었으나, 원격 탐사 이미지(RS images)에 적용하는 연구는 아직 초기 단계이다. 기존 연구인 LGCE나 RMSIN 등은 LAVT(Language-aware Vision Transformer) 프레임워크를 기반으로 하여 작은 객체나 회전된 객체를 분할하기 위한 인터랙션 네트워크를 설계하였다. 그러나 이러한 단일 단계(single-stage) 방식들은 여전히 레이블링된 데이터의 부족이라는 한계에 직면해 있다.

### Visual Grounding on RS Images

RS 이미지의 Visual Grounding은 객체 간의 지리적 관계 파악이 중요하다. GeoVG, MGVLF, LQVG 등의 모델이 제안되었으며, 본 논문에서는 Grounding 성능이 우수한 LQVG를 좌표 프롬프트 생성 모델로 채택하여 사용한다.

기존 RIS 방식들이 엔드-투-엔드(end-to-end) 세그멘테이션 방식을 취한 것과 달리, 본 논문은 SAM과 Visual Grounding 모델을 결합하여 학습 데이터 부담을 획기적으로 줄이는 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인

PSLG-SAM은 입력 이미지 $I \in \mathbb{R}^{H \times W \times 3}$와 텍스트 표현 $T \in \mathbb{R}^{1 \times d}$를 받아 다음의 과정을 거친다.

1. **Visual Grounding 단계**: LQVG 모델을 사용하여 텍스트와 이미지의 특징을 융합하고 타겟 객체의 바운딩 박스 좌표 $F_C$를 생성한다.
    $$F_C = \text{Dec}_{vg}(F_v \odot \phi(F_t))$$
    여기서 $F_v$와 $F_t$는 각각 시각 및 텍스트 특징이며, $\odot$는 융합 연산, $\phi$는 매핑 연산을 의미한다.

2. **SAM 기반 정밀 분할 단계**:
    - **Frozen Visual Encoder**: ViT 구조의 SAM 인코더를 동결(frozen) 상태로 사용하여 글로벌 이미지 특징 $E_I$를 추출한다.
    - **Prompt Encoder**: 앞서 얻은 바운딩 박스 $F_C$와 CFPG 모듈에서 생성된 포인트 프롬프트 $D(x,y)$를 입력으로 받아 프롬프트 특징 $E_p$를 생성한다.
      $$E_p = \text{Enc}_p(F_C, D(x,y))$$
    - **Mask Decoder**: $E_I$와 $E_p$를 결합하여 초기 마스크 $F_o$를 생성한다.
      $$F_o = \text{Dec}_{SAM}(E_I, E_p)$$

### CFPG (Clustering-based Foreground Point Generator)

바운딩 박스의 부정확성을 해결하기 위해 전경 포인트 프롬프트를 생성하는 과정이다.

1. **ROI 추출 및 클러스터링**: 바운딩 박스 영역을 크롭하여 $k=2$인 KMeans++ 클러스터링을 수행하여 전경과 배경을 일차적으로 구분한다.
2. **전경 정제**: Distance Transform과 Watershed 알고리즘을 적용하여 연결된 영역들을 분리한다.
3. **최적 영역 선택**: 각 연결 영역 $R_i$의 면적 $A(R_i)$와 볼록성(convexity) $\kappa_i$를 계산한다.
    $$\kappa_i = \frac{A(R_i)}{A_{hull}(R_i)}$$
    면적 임계값을 만족하는 영역 중 볼록성이 가장 높은 영역 $R_{best}$를 최종 전경으로 선택한다.
4. **중심점 계산**: $R_{best}$의 기하학적 중심 $(x_c, y_c)$를 계산하여 SAM의 포인트 프롬프트로 사용한다.

### MBO (Mask Boundary Iterative Optimization)

생성된 마스크의 경계를 정교화하는 전략이다.

1. **GMM 모델링**: 마스크의 침식(erosion) 연산을 통해 얻은 전경/배경 영역으로 각각의 Gaussian Mixture Model(GMM)을 구축하여 픽셀 $Z_i$가 전경일 확률 $P_F(Z_i)$와 배경일 확률 $P_B(Z_i)$를 계산한다.
2. **에너지 최소화**: 데이터 항(data term, 픽셀-모델 일치도)과 평활도 항(smoothness term, 인접 픽셀 간 유사도)의 합으로 구성된 목적 함수를 정의한다.
3. **최적화**: Max-Flow Min-Cut 알고리즘을 통해 픽셀 레이블을 최적화하고, EM 알고리즘으로 GMM 파라미터를 업데이트하는 과정을 수렴할 때까지 반복한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 기존의 RRSIS-D와 본 논문에서 제안한 RRSIS-M을 사용하였다. RRSIS-M은 5,465장의 고해상도 이미지와 20개 카테고리로 구성되어 있으며, 전문가의 정밀한 주석을 포함한다.
- **지표**: Overall IoU (oIoU), Mean IoU (mIoU), Precision@X (P@X)를 사용하였다.
- **비교 대상**: RRN, CMSA, LSCM, CMPC, BRINet, LGCE, LAVT, RMSIN 등과 성능을 비교하였다.

### 정량적 결과

- **RRSIS-D 데이터셋**:
  - 약하게 지도된(weakly supervised, 바운딩 박스만 사용) 설정에서도 mIoU 기준 기존 SOTA인 RMSIN보다 $3.2\%$ 향상된 성능을 보였다.
  - 마스크 레벨 데이터로 디코더를 미세 조정(fine-tuning)했을 때는 mIoU가 $6.4\%$ 향상되어 모든 지표에서 SOTA를 달성하였다.
- **RRSIS-M 데이터셋**: 기존 방식 대비 mIoU $3\%$, oIoU $1.3\%$ 향상을 기록하였다.

### 절제 연구 (Ablation Study)

- **구성 요소 효과**: CFPG 모듈은 P@X와 mIoU를 크게 향상시켰으며, MBO 전략은 oIoU를 유의미하게 높였다. 두 모듈을 모두 사용했을 때 oIoU $5.6\%$, mIoU $2.1\%$의 시너지 효과가 나타났다.
- **데이터 효율성**: 마스크 데이터의 $10\%$만 사용하여 디코더를 학습시켜도 상당한 성능 향상이 있었으며, $100\%$를 사용했을 때와 성능 차이가 크지 않았다. 이는 SAM 디코더의 파라미터 수가 적어 적은 데이터로도 도메인 적응이 가능함을 시사한다.

## 🧠 Insights & Discussion

본 논문은 RRSIS 작업을 위치 추정과 분할로 분리함으로써, 원격 탐사 이미지의 복잡한 배경으로 인한 간섭을 효과적으로 회피하였다. 특히, t-SNE 시각화 분석을 통해 SAM 인코더가 기존 RMSIN의 인코더보다 클래스 간 거리는 멀고 클래스 내 밀도는 높게 유지함을 확인하였으며, 이는 SAM이 매우 정교한 세밀 식별 능력(fine-grained discriminative ability)을 갖추고 있음을 입증한다.

한계점으로는, 약하게 지도된 설정에서 oIoU가 상대적으로 낮게 나타나는데, 이는 마스크 데이터 없이 바운딩 박스만으로는 미지의 원격 탐사 이미지에서 타겟과 배경의 경계를 완벽하게 구분하는 데 한계가 있기 때문으로 분석된다. 또한, MBO 전략이 반복적 최적화 과정을 거치므로 추론 속도에 영향을 줄 수 있다는 점이 고려되어야 한다.

그럼에도 불구하고, 매우 적은 양의 마스크 데이터만으로도 높은 성능을 낼 수 있다는 결과는 주석 비용이 매우 높은 원격 탐사 분야에서 본 프레임워크가 매우 실용적임을 보여준다.

## 📌 TL;DR

본 논문은 원격 탐사 이미지의 참조 분할(RRSIS) 문제를 해결하기 위해 **위치 추정(Visual Grounding) $\rightarrow$ 정밀 분할(SAM)**의 2단계 프레임워크인 **PSLG-SAM**을 제안하였다. 특히 비지도 학습 기반의 **CFPG**를 통해 정확한 포인트 프롬프트를 생성하고, **MBO** 전략으로 경계선을 최적화하여 정밀도를 높였다. 또한 고품질의 **RRSIS-M 데이터셋**을 함께 제공하였다. 실험 결과, 적은 데이터만으로도 기존 SOTA 모델을 능가하는 성능을 보였으며, 이는 데이터 부족 문제가 심한 원격 탐사 이미지 분석 분야에 효율적인 해결책이 될 수 있다.
