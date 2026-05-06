# SURGICAL SCENE SEGMENTATION BY TRANSFORMER WITH ASYMMETRIC FEATURE ENHANCEMENT

Cheng Yuan, Yutong Ban (2024)

## 🧩 Problem to Solve

본 논문은 로봇 보조 복강경 수술(robotic-assisted laparoscopic surgery)의 이해를 위한 핵심 과제인 수술 장면 분할(Surgical scene segmentation) 문제를 해결하고자 한다. 수술 장면 분할은 다양한 해부학적 구조와 수술 도구를 정확하게 구분하는 것이 필수적이지만, 다음과 같은 두 가지 주요 어려움이 존재한다.

첫째, 수술 장면 내의 서로 다른 해부학적 구조와 도구들이 매우 유사한 지역적 텍스처(local textures)를 가지고 있어 오인식 가능성이 높다. 둘째, 가느다란 도구나 실(thread)과 같은 세밀한 구조(fine-grained structures)들이 복잡하게 얽혀 있어 경계가 모호하고 굴곡이 심해 정확한 마스크 생성이 어렵다.

기존의 Transformer 기반 방법론들은 전역적 문맥 파악에는 능숙하지만, 패치 내부의 정보 융합(inner-patch information fusion)이 부족하고 해부학적 구조와 도구의 특수성을 모델링하지 못한다는 한계가 있다. 따라서 본 논문의 목표는 이러한 지역적 유사성을 극복하고 세밀한 구조 인식 능력을 향상시킨 Transformer 기반의 분할 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Transformer의 전역적 표현 능력과 Convolution의 지역적 특징 추출 능력을 결합하되, 특히 수술 장면의 기하학적 특성을 반영한 **Asymmetric Feature Enhancement (AFE)** 모듈을 도입하는 것이다.

주요 기여 사항은 다음과 같다.

1. **Multi-scale Interaction Attention (MIA)** 구조를 통해 Convolution 기반의 강화된 특징 피라미드를 Transformer 인코더-디코더 백본에 융합함으로써, 기존 pre-trained 가중치를 유지하면서도 지역적 특징 유사성 문제를 해결하였다.
2. **Asymmetric Feature Enhancement (AFE)** 모듈을 설계하여 해부학적 구조(다각형 형태)와 수술 도구(바 또는 튜브 형태)의 서로 다른 특성을 각각 symmetric 및 asymmetric convolution으로 포착하도록 하였다.
3. 제안한 TAFE 방법론이 Endoscapes2023 및 EndoVis2018 데이터셋에서 기존 SOTA 방법론들을 뛰어넘는 성능을 보였으며, 특히 미세 구조 인식 능력이 크게 향상됨을 입증하였다.

## 📎 Related Works

수술 장면 분할을 위한 기존 연구는 크게 두 가지 방향으로 나뉜다.

1. **Convolutional Neural Networks (CNN) 기반 방법**: Atrous convolution을 사용하여 다중 스케일 문맥 특징을 얻거나, low-rank bilinear feature fusion 등을 통해 인식 능력을 높이려는 시도가 있었다. 최근에는 Long Strip Kernel Attention Network (LSKANet) 등이 우수한 성과를 거두었다.
2. **Transformer 기반 방법**: Mask2Former나 MaskDINO와 같이 쿼리 기반(query-based)의 Transformer 구조를 사용하여 밀집 예측(dense prediction)을 수행하는 연구들이 진행되었다.

그러나 이러한 기존 방식들은 수술 장면 특유의 지역적 특징 유사성(local feature similarity)과 복잡한 교차 구조(intersected structure complexity) 문제를 충분히 다루지 못했다. 이로 인해 튜브 형태의 도구를 인식하는 데 어려움을 겪거나 마스크의 경계가 거칠게 생성되는 문제가 발생하였다.

## 🛠️ Methodology

### 전체 시스템 구조

제안된 TAFE 프레임워크는 세 가지 주요 구성 요소로 이루어져 있다.

- **Transformer Encoder-Decoder Backbone**: 기본 뼈대로 사용되며, 오픈 소스의 pre-trained 가중치를 그대로 활용할 수 있다.
- **Multi-scale Interaction Attention (MIA) Branch**: Convolution을 통해 추출된 4단계(1/4, 1/8, 1/16, 1/32 해상도)의 특징 피라미드를 Transformer의 임베딩과 상호 융합한다.
- **Asymmetric Feature Enhancement (AFE) Module**: 특징 피라미드의 각 층에서 해부학적 구조와 도구의 특징을 강화한다.

### Multi-scale Interaction Attention (MIA)

Transformer 인코더의 특징 $F$와 AFE 모듈에 의해 강화된 다중 스케일 Convolution 특징 피라미드 $\{E_1, E_2, E_3, E_4\}$를 융합한다. 먼저 $F$를 4개의 레이어로 unflatten 하여 다음과 같이 픽셀 값을 계산한다.

$$F_{UF}^l(i, j) = F_{i + (j-1) * R^l + \sum_{k=0}^{l-1} R^2_k}$$

여기서 $l \in \{1, 2, 3, 4\}$이며, $R^l$은 피라미드의 $l$번째 레이어 특징 맵의 크기이다. 이렇게 변환된 $F_{UF}^l$은 강화된 특징 맵 $E^l$과 element-wise addition을 통해 합쳐진 후 다음 AFE 모듈로 전달된다.

### Asymmetric Feature Enhancement (AFE)

AFE 모듈은 해부학적 구조와 도구의 시각적 특성이 다르다는 점에 착안하여 두 개의 블록으로 구성된다.

1. **Anatomy Enhancement Block**: 불규칙한 다각형 형태의 해부학적 구조를 인식하기 위해 설계되었다. $5 \times 5$ 표준 convolution으로 구조 정보를 집계한 후, 세 개의 **symmetric convolution** 브랜치를 사용한다. 구현 시 계산 효율을 위해 다음과 같은 cascaded strip convolution pair를 사용한다.
   $$S_{AE}^{l,m} = \text{Conv}_{k_m \times 1} (\text{Conv}_{1 \times k_m} (C_{AE}^l))$$
   이때 커널 크기 $k_m$은 $\{3, 5, 7\}$로 설정된다.

2. **Instrument Enhancement Block**: 정형화된 바(bar) 또는 튜브 형태의 도구를 인식하기 위해 설계되었다. 수직 및 수평 차원의 특징을 동시에 학습하기 위해 **parallel strip convolution** 구조를 사용한다.
   $$S_{IE}^{l,m} = \text{Conv}_{k_m \times 1} (C_{IE}^l) + \text{Conv}_{1 \times k_m} (C_{IE}^l)$$
   마찬가지로 $k_m \in \{3, 5, 7\}$이다.

최종적으로, 각 블록에서 나온 특징 맵을 합산하고 shortcut connection을 추가한 뒤 $1 \times 1$ convolution을 거쳐 attention map $E^l$을 생성하며, 이를 집계된 특징 맵 $C_E^l$과 곱한다.

$$E^l = \left( \text{Conv}_{1 \times 1} \left( \sum_{m=0}^2 S_E^{l,m} + C_E^l \right) \right) \otimes C_E^l$$

## 📊 Results

### 실험 설정

- **데이터셋**: Endoscapes2023(담낭절제술 비디오 50개, 493프레임) 및 EndoVis2018(19개 시퀀스)을 사용하였다.
- **평가 지표**: Endoscapes2023에서는 mAP@[0.5:0.95]를, EndoVis2018에서는 mIoU와 mDice를 사용하였다.
- **비교 대상**: Mask-RCNN, Mask2Former, MaskDINO, LSKANet, SegFormer 등 최신 SOTA 모델들과 비교하였다.

### 정량적 결과

- **Endoscapes2023**: 전체 detection mAP 32.5%, segmentation mAP 30.6%를 기록하며 최고 성능을 달성하였다. 특히 인식이 어려운 'hepatocystic triangle dissection'과 'cystic artery' 항목에서 각각 19.2%와 4.3%의 segmentation mAP 향상을 보였다.
- **EndoVis2018**: mIoU 77.5%, mDice 86.6%로 가장 우수한 성과를 거두었다. 이전 SOTA인 LSKANet 대비 성능 향상이 뚜렷하며, 특히 일부 시퀀스(Seq 2, 4)에서는 mIoU가 각각 19.3%, 23.5% 크게 향상되었다.

### 정성적 결과 및 분석

시각적 비교 결과, TAFE는 다음과 같은 강점을 보였다.

- **미세 구조 인식**: 다른 모델들이 놓치기 쉬운 담낭동맥(cystic artery)이나 담낭판(cystic plate)과 같은 세밀한 구조를 정확하게 식별하였다.
- **연속성 유지**: 가느다란 실(thread)과 같은 튜브 형태의 객체에 대해 끊김 없이 연속적이고 완전한 분할 결과를 생성하였다.
- **경계 구분**: 신장 실질(kidney parenchyma)과 피복 신장(cover kidney)처럼 경계가 모호한 영역에서도 더 명확하고 정확한 분할 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 Transformer의 전역적 문맥 파악 능력과 CNN의 지역적 특징 추출 능력을 결합하는 단순한 융합을 넘어, 수술 도구와 장기의 **기하학적 특성(Anisotropy)**을 고려한 비대칭적 설계(Asymmetric Design)를 도입했다는 점에서 큰 의의가 있다.

특히, $\text{Conv}_{k \times 1}$과 $\text{Conv}_{1 \times k}$를 조합한 strip convolution을 통해 연산량을 줄이면서도 바(bar) 형태의 도구 특성을 효과적으로 포착한 점이 인상적이다. 또한, Transformer 백본을 수정하지 않고 MIA 브랜치를 통해 특징을 주입함으로써 기존의 강력한 pre-trained 가중치를 그대로 활용할 수 있게 설계하여 학습 효율성과 성능을 동시에 잡았다.

다만, 본 논문에서 제안한 AFE 모듈의 커널 크기($3, 5, 7$) 선정 근거에 대한 상세한 분석은 명시되지 않았으며, 다양한 수술 환경(조명 변화, 출혈로 인한 가림 현상 등)에서의 강건성에 대한 추가적인 논의가 이루어졌다면 더욱 완성도 높은 연구가 되었을 것이다.

## 📌 TL;DR

본 논문은 수술 장면의 지역적 텍스처 유사성과 세밀한 구조 인식의 어려움을 해결하기 위해, Transformer 백본에 **Multi-scale Interaction Attention (MIA)**와 **Asymmetric Feature Enhancement (AFE)** 모듈을 결합한 **TAFE** 프레임워크를 제안한다. AFE 모듈은 해부학적 구조와 수술 도구의 형태적 특성을 구분하여 처리함으로써, 기존 SOTA 모델 대비 미세 구조(실, 혈관 등) 인식 능력을 획기적으로 향상시켰으며, EndoVis2018 및 Endoscapes2023 데이터셋에서 최고 성능을 달성하였다. 이 연구는 향후 수술 내비게이션 및 위험 평가와 같은 고차원적인 수술 이해 작업의 기초가 될 가능성이 높다.
