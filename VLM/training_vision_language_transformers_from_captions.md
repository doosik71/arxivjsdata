# Training Vision-Language Transformers from Captions

Liangke Gui, Yingshan Chang, Qiuyuan Huang, Subhojit Som, Alex Hauptmann, Jianfeng Gao, Yonatan Bisk (2023)

## 🧩 Problem to Solve

본 논문은 기존의 Vision-Language (VL) Transformer 모델들이 시각적 특징을 추출하기 위해 의존해온 고비용의 지도 학습 기반 사전 훈련 방식과 그로 인한 제약 사항을 해결하고자 한다. 기존의 많은 모델들은 Object Detection을 위해 ImageNet과 같은 대규모 데이터셋의 클래스 라벨이나 Bounding Box(BBox) 정보를 활용하여 visual backbone을 먼저 학습시킨 후, 이를 다중 모달 파이프라인에 통합하는 방식을 사용했다.

이러한 방식은 두 가지 주요 문제를 야기한다. 첫째, Region of Interest (ROI) 추출 과정이 계산적으로 매우 비싸며 추론 속도를 저하시킨다. 둘째, ImageNet과 같은 특정 라벨 공간에 기반한 사전 훈련은 모델의 표현력을 해당 라벨 집합 내로 제한하여, 학습 데이터에 포함되지 않은 새로운 시각적 범주(Open-vocabulary)에 대한 일반화 능력을 떨어뜨린다. 따라서 본 연구의 목표는 클래스 라벨이나 BBox 같은 정교한 인간의 주석(human labels) 없이, 오직 이미지-캡션 쌍(image-caption pairs)만을 활용하여 효율적이고 일반화 성능이 뛰어난 VL Transformer 모델인 VLC(Vision-Language from Captions)를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 설계 아이디어는 지도 학습 기반의 사전 훈련을 완전히 제거하고, **Masked Image Modeling (MIM)**을 통해 시각적 의미론을 스스로 학습하게 하는 것이다.

구체적으로, 본 연구는 Masked Auto-Encoders (MAE)를 사용하여 라벨이 없는 ImageNet-1K 데이터셋으로 초기화함으로써, 특정 클래스에 편향되지 않은 유연한 시각적 표현 공간을 확보하였다. 또한, 텍스트 토큰뿐만 아니라 이미지 패치를 동시에 마스킹하고 이를 복원하는 MIM 목적 함수를 도입하여, 시각적 표현과 언어적 표현이 상호 보완적으로 정렬(alignment)되도록 설계하였다. 이를 통해 ROI 추출 과정 없이도 픽셀 수준의 세밀한 정렬이 가능함을 보여주었으며, 이는 결과적으로 모델의 크기를 줄이면서도 추론 속도를 높이고, 정교한 주석 없이도 높은 성능을 달성하는 결과로 이어졌다.

## 📎 Related Works

기존의 VL 모델링 연구는 이미지 인코딩 방식에 따라 크게 세 가지 범주로 나뉜다.
첫째는 Faster R-CNN과 같은 사전 훈련된 객체 검출기를 사용하여 영역 수준의 특징을 추출하는 방식(예: UNITER, OSCAR, VinVL)이다. 이들은 높은 성능을 보이지만, ROI 추출 과정이 매우 느리고 고해상도 입력이 필요하다는 한계가 있다.
둘째는 CNN의 그리드 특징을 사용하는 방식(예: SOHO)으로, 계산 효율성을 높이려 했으나 여전히 CNN 기반의 제약이 존재한다.
셋째는 Vision Transformer (ViT)를 인코더로 사용하는 방식(예: ViLT)이다. ViLT는 ROI 추출 과정을 제거하여 효율성을 극대화했지만, 여전히 ImageNet-21K와 같은 지도 학습된 가중치로 초기화해야 한다는 점이 한계로 지적된다.

본 논문의 VLC는 이러한 기존 방식들과 달리, 지도 학습된 가중치나 객체 검출기 없이 MAE와 같은 자기지도 학습(Self-supervised learning) 기반의 초기화와 MIM-MLM-ITM의 통합 목적 함수를 사용함으로써 차별점을 가진다. 특히 기존 연구들이 MIM이 downstream task 성능 향상에 기여하지 않거나 오히려 방해한다고 주장한 것과 달리, VLC는 충분한 학습 단계가 확보될 때 MIM이 일관된 성능 향상을 가져온다는 점을 입증하였다.

## 🛠️ Methodology

### 전체 시스템 구조

VLC 모델은 크게 세 가지 모듈로 구성된다.

1. **Modality-specific Projection Module**: 이미지 패치와 텍스트 토큰을 동일한 차원의 임베딩 공간으로 투영하는 단순 선형 층이다.
2. **Multi-modal Encoder**: MAE로 초기화된 12층 구조의 ViT-B/16 또는 ViT-L/16 백본을 사용하며, 시각과 언어 모달리티를 동시에 입력받는 Single-stream 구조를 취한다.
3. **Task-specific Decoder**: 사전 훈련 단계에서 MIM, MLM, ITM 목적 함수를 수행하기 위한 헤드들로 구성된다.

### 상세 방법 및 방정식

#### 1. 모달리티 투영 (Projection)

이미지 패치 $v$와 텍스트 토큰 $w$는 각각 위치 임베딩($v^{pos}, w^{pos}$)과 모달리티 타입 임베딩($v^{type}, w^{type}$)이 더해진 후 LayerNorm을 거쳐 최종 표현 $\hat{v}_i, \hat{w}_j$가 된다.
$$\hat{v}_i = \text{LayerNorm}(v_i + v^{pos}_i + v^{type})$$
$$\hat{w}_j = \text{LayerNorm}(w_j + w^{pos}_j + w^{type})$$

#### 2. 사전 훈련 목적 함수 (Pretraining Objectives)

모델은 다음 세 가지 손실 함수의 합으로 학습된다: $L = L_{MLM} + L_{ITM} + L_{MIM}$.

- **Masked Language Modeling (MLM)**: 텍스트 토큰의 15%를 `[MASK]`로 치환하고, 마스킹되지 않은 텍스트($w_{\setminus m}$)와 이미지 패치($v_{\setminus m}$)를 통해 원래 토큰($w_m$)을 예측한다.
$$L_{MLM} = -\mathbb{E}_{(w,v) \sim D} \log p(w_m | w_{\setminus m}, v_{\setminus m})$$

- **Masked Image Modeling (MIM)**: 이미지 패치의 60%를 마스킹하고, 나머지 토큰과 패치를 이용하여 마스킹된 영역의 원본 픽셀 값을 직접 회귀(regression) 방식으로 복원한다. 8층의 Transformer를 MIM 헤드로 사용하여 픽셀 값을 예측하며, 손실 함수는 다음과 같다.
$$L_{MIM} = \mathbb{E}_{(w,v) \sim D} \|r(h^v_i) - v_i\|^2$$
여기서 $r$은 MIM 헤드, $h^v_i$는 인코더의 출력값, $v_i$는 원본 픽셀 값이다.

- **Image-Text Matching (ITM)**: 이미지와 텍스트 쌍이 서로 일치하는지 여부를 이진 분류한다. `[CLS]` 토큰의 최종 표현 $h_{CLS}$를 입력으로 사용한다.
$$L_{ITM} = -\mathbb{E}_{(w,v) \sim D} \log p(y | w, v)$$
여기서 $y \in \{0, 1\}$은 매칭 여부를 나타낸다.

### 추론 절차: BBox 생성 (PUSH 알고리즘)

VLC는 BBox를 직접 예측하는 헤드가 없으므로, 패치와 텍스트 간의 코사인 유사도를 기반으로 하는 **Affinity Map**을 생성한다.
$$\hat{A}_{t,p} = \cos(L_t, V_p)$$
이후 **PUSH 알고리즘**을 통해 이 맵에서 가장 적절한 사각형 영역을 탐색한다. 전체 이미지에서 시작하여 사각형의 네 면을 조금씩 안으로 밀어 넣으며(push), $M$이라는 기준(IoU와 유사한 지표)이 더 이상 개선되지 않을 때까지 반복하여 최종 BBox를 결정한다.

## 📊 Results

### 실험 설정

- **데이터셋**: COCO, Visual Genome, GCC, SBU Captions (총 4.0M 이미지, 5.1M 쌍) 및 VinVL 데이터셋(5.6M 이미지)을 사용하였다.
- **평가 지표**: Image-Text Retrieval (Recall@K), VQAv2 (Accuracy), NLVR2 (Accuracy), Refcoco (IoU acc) 등을 사용하였다.

### 주요 결과

1. **Image-Text Retrieval**: $VLC_{Base}$는 동일 규모의 ViLT보다 우수한 성능을 보였으며, $VLC_{Large}$는 ROI 기반의 고성능 모델인 ALBEF 및 UNITER Large와 경쟁 가능한 수준의 성능을 달성하였다.
2. **Image-Text Understanding**: VQAv2와 NLVR2에서 $VLC_{Large}$는 지도 학습 기반의 VinVL과 유사하거나 더 높은 성능을 보였다. 특히, 지도 학습된 모델보다 모델 크기를 키웠을 때의 성능 향상 폭이 훨씬 컸다.
3. **Image-Text Grounding**: Refcoco 등에서 기존의 모듈형 모델(TransVG 등)을 압도하였으며, 훨씬 가벼운 파라미터 수와 빠른 추론 속도로도 통합 모델(OFA 등)에 근접하는 성능을 냈다.
4. **Zero-shot 일반화**: 추상적 시각 추론 데이터셋인 Kilogram에서 ViLT보다 일관되게 높은 정확도를 기록하여, ImageNet의 사전 지식에 얽매이지 않은 일반화 능력을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **MIM의 재발견**: 기존 연구들과 달리 MIM이 학습 단계가 길어질수록 downstream task 성능을 지속적으로 향상시킨다는 것을 발견하였다. 이는 픽셀 수준의 복원 작업이 모델로 하여금 더 세밀한 시각적 특징을 학습하게 하기 때문이다.
- **사전 지식의 제거**: ImageNet의 지도 학습된 가중치를 사용하지 않음으로써, 모델이 특정 클래스에 편향되지 않고 텍스트-이미지 간의 더 유연하고 일반적인 정렬을 학습할 수 있었다. 이는 시각화 결과에서도 VLC가 ViLT보다 더 세분화되고 확산된(diffuse) 패치 표현을 가지는 것으로 나타났다.
- **효율성**: ROI 추출 과정을 제거하고 단순한 linear projection만을 사용함으로써 추론 속도를 획기적으로 개선하였다.

### 한계 및 비판적 해석

- **복잡한 관계 추론의 어려움**: Grounding 실험에서 대상 객체가 매우 작거나, "보라색 셔츠를 입은 여자 옆의 남자"와 같이 복잡한 관계 기반의 속성으로 정의될 때 예측 정확도가 떨어지는 경향을 보였다. 이는 단순한 정렬(alignment)을 넘어 고차원적인 공간 추론 능력이 여전히 부족함을 시사한다.
- **데이터 의존성**: 성능이 데이터 규모와 모델 크기에 따라 계속해서 증가하는 경향을 보이므로, 향후 더 거대한 웹 데이터셋을 활용한 확장 가능성은 높으나, 이에 따른 컴퓨팅 자원 소모가 클 것으로 예상된다.

## 📌 TL;DR

본 논문은 클래스 라벨이나 Bounding Box 같은 지도 학습 데이터 없이, 오직 이미지-캡션 쌍만을 이용해 학습하는 **VLC (Vision-Language from Captions)** 모델을 제안한다. MAE 초기화와 MIM, MLM, ITM의 통합 학습 전략을 통해, 기존의 ROI 기반 모델이나 지도 학습 기반 ViT 모델보다 효율적이면서도 강력한 일반화 성능을 달성하였다. 특히 MIM의 중요성을 재입증하고, 단순한 정렬 맵으로부터 BBox를 추출하는 PUSH 알고리즘을 통해 높은 효율성의 시각적 접지(Grounding)를 구현하였다. 이 연구는 향후 대규모의 약지도 학습(weakly-supervised) 데이터를 이용한 오픈 보캐블러리 VL 모델 연구에 중요한 기반을 제공할 것으로 보인다.
