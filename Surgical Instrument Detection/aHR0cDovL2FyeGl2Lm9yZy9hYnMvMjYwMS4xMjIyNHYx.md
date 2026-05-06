# Where It Moves, It Matters: Referring Surgical Instrument Segmentation via Motion

Meng Wei et al. (2026)

## 🧩 Problem to Solve

본 논문은 자연어 기술을 통해 수술 비디오 내의 특정 수술 도구를 식별하고 분할하는 **Referring Surgical Instrument Segmentation (RSIS)** 문제를 해결하고자 한다.

수술 장면에서 도구를 정밀하게 분할하는 것은 지능형 수술실 구현, 자율 수술 로봇 보조, 수술 교육 및 스킬 평가를 위해 매우 중요하다. 그러나 기존의 Referring Segmentation 방식들은 주로 도구의 외형(Appearance)이나 사전 정의된 도구 이름과 같은 **정적 시각적 단서(Static visual cues)**에 의존하는 경향이 있다. 이러한 방식은 다음과 같은 수술 환경의 특수성으로 인해 일반화 성능이 떨어진다는 문제가 있다.

1. **시각적 모호성**: 수술 도구 간의 외형이 매우 유사하며, 출혈, 연기, 조명 부족 및 도구 간의 가려짐(Occlusion)이 빈번하게 발생한다.
2. **언어적 비표준화**: 동일한 도구라도 병원이나 지역, 수술 프로토콜에 따라 부르는 명칭이 달라 표준화된 언어 표현을 사용하기 어렵다.

따라서 본 연구의 목표는 외형보다는 도구의 **움직임(Motion)**, 즉 도구가 어떻게 움직이고 상호작용하는지에 집중하여, 가려짐이나 모호한 용어 환경에서도 강건하게 도구를 분할할 수 있는 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 수술 도구의 외형이 아닌 **동적인 움직임 패턴(Motion-centric signals)**을 언어 표현과 결합하여 도구를 식별하는 것이다. 수술 절차는 도구의 진입 경로, 견인 패턴, 조직과의 상호작용 등 일련의 정형화된 움직임으로 정의되므로, 이는 외형보다 더 일관적이고 해석 가능한 신호를 제공한다.

주요 기여 사항은 다음과 같다.

- **Ref-IMotion 데이터셋 구축**: 4개의 공개 데이터셋(EndoVis-17, EndoVis-18, CholecSeg8k, GraSP)을 통합하고, 도구의 궤적, 방향, 상호작용을 기술하는 **움직임 중심의 Referring Expression**을 수동으로 주석 달아 구축하였다.
- **SurgRef 프레임워크 제안**: 정적 표현과 움직임 중심 표현 모두를 처리할 수 있는 motion-guided referring video segmentation 프레임워크를 제안하였다.
- **Key-frame Attention 모듈**: 언어 가이드 기반의 객체 세만틱을 활용해 비디오 내에서 의미 있게 정렬된 프레임만을 적응적으로 선택하여 연산 효율성과 분할 정밀도를 높였다.

## 📎 Related Works

### 기존 연구 및 한계

1. **Video Segmentation**: STM, XMem 등 메모리 기반 아키텍처를 통해 시간적 일관성을 유지하려는 시도가 있었으나, 이는 주로 일반 영상 도메인에 집중되어 있다.
2. **Surgical Video Segmentation**: LWANet, MATIS 등 픽셀 수준의 지도 학습을 통한 분할 연구가 활발했으나, 이는 고정된 클래스에 대한 분할일 뿐 사용자의 자연어 쿼리에 대응하는 Referring Segmentation과는 거리가 있다.
3. **Surgical Referring Segmentation**: VIS-Net, TP-SIS 등이 제안되었으나, 이들은 여전히 정적인 외형 단서나 명시적인 도구 이름에 과도하게 의존하며, 장기적인 시간적 모델링(Long-term temporal modeling)이 부족하여 복잡한 수술 시나리오에서 한계를 보인다.

### 차별점

SurgRef는 기존 방식들이 간과했던 **움직임(Motion)**이라는 변수를 명시적으로 모델링한다. 특히, "오른쪽에서 진입하여 담낭을 중앙으로 견인하는 도구"와 같이 시간적 흐름과 궤적이 포함된 동적 표현을 이해함으로써, 외형이 유사한 도구가 동시에 존재하거나 일부가 가려진 상황에서도 정확한 타겟팅이 가능하다.

## 🛠️ Methodology

### 1. Ref-IMotion 데이터셋 구축

다양한 수술 절차와 도구 셋을 포괄하기 위해 다음과 같이 구성하였다.

- **구성**: EndoVis-IM17, EndoVis-IM18, CholecSeg8k-IM, GraSP-IM.
- **주석 내용**: 단순한 도구 이름이 아니라 $\text{궤적} \rightarrow \text{방향} \rightarrow \text{상호작용}$을 포함하는 표현을 생성하였다. (예: "Grasper enters from the left and retracts the gallbladder medially")
- **데이터 규모**: 총 319개의 비디오 클립, 21,350 프레임, 718개의 Referring Expression(그 중 358개가 움직임 기반)을 포함한다.

### 2. SurgRef 아키텍처

전체 시스템은 시각 특징 추출기, 텍스트 인코더, 그리고 이를 통합하는 Transformer Decoder로 구성된다.

- **시각 및 텍스트 인코딩**: 시각 특징 $F^{image}$는 Swin Transformer를 통해 추출하며, 텍스트 표현 $F^{text}$는 frozen RoBERTa-base 모델을 통해 고정된 세만틱 임베딩으로 변환한다.
- **Language-driven Queries**: Mask2Former의 일반적인 쿼리 대신, 텍스트 임베딩을 주입한 초기 쿼리를 생성한다.
  $$q^{(0)}_i = W_{init} \cdot F_{text} + b_i$$
  여기서 $W_{init}$은 학습 가능한 투영 행렬이며, $b_i$는 위치 편향(positional bias)이다.
- **분할 헤드(Heads)**:
  - **Classification Head**: 쿼리가 타겟 객체와 일치하는지 점수를 계산한다.
      $$\hat{y}_{cls}^i = \text{Linear}([h_{q(L)}^i \parallel F_{text}])$$
  - **Mask Embedding Head**: 쿼리를 마스크 임베딩 $e_i$로 변환하고, 픽셀 특징 $F_{mask}$와의 내적을 통해 이진 마스크 $\hat{M}_i$를 생성한다.
      $$\hat{M}_i = \sigma(e_i^\top \cdot F_{mask})$$

### 3. Key-frame Selection (KFS) 및 Inter-frame Attention

수술 비디오의 시간적 중복성을 줄이기 위해 도입된 모듈이다.

- **Relevance Score 계산**: 각 프레임 $t$에 대해 언어-시각 정렬 점수 $s_t$를 계산한다.
  $$s_t = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot e_t) + b)$$
- **프레임 선택**: 점수가 가장 높은 상위 $T'$개의 프레임(본 논문에서는 $T'=8$)만을 선택하여 디코더에 전달한다.
- **Inter-frame Attention**: 선택된 키 프레임들 간에 Multi-head Self-Attention을 적용하여 시간적 차원의 정보를 교환함으로써 객체의 움직임을 캡처한다.

### 4. 학습 절차 및 손실 함수

모델은 프레임 수준의 손실(Cross-Entropy, Binary Cross-Entropy, Dice Loss)과 비디오 수준의 손실(인접 프레임 간 쿼리 임베딩의 일관성을 강제하는 Temporal Similarity Loss)의 합으로 학습된다.

## 📊 Results

### 실험 설정

- **데이터셋**: EndoVis-IM17, EndoVis-IM18, GraSP-IM (학습 및 테스트), CholecSeg8k-IM (Zero-shot 테스트).
- **지표**: J (Region Similarity), F (Contour Accuracy), J&F (평균), Dice, IoU.
- **비교 대상**: VIS-Net, VISA, MPG-SAM 2.

### 주요 결과

1. **성능 우위**: SurgRef는 모든 데이터셋에서 SOTA 성능을 달성하였다. 특히 EndoVis-IM17에서 J&F 기준 88.93을 기록하여 MPG-SAM 2(84.37)보다 우수한 성능을 보였다.
2. **Zero-shot 일반화**: 로봇 보조 전립선 절제술(GraSP-IM) 데이터로 학습한 모델이 복강경 담낭 절제술(CholecSeg8k-IM) 데이터에서 별도의 튜닝 없이도 준수한 성능을 보였다. 이는 외형이 아닌 '움직임'이라는 보편적 신호를 학습했기 때문이다.
3. **움직임 표현의 효과**: 움직임 기반 표현으로 학습한 모델이 정적 표현(외형, 공간 정보)으로 테스트했을 때조차 더 높은 성능을 보였다. (예: EndoVis-IM17에서 J&F 89.42 vs 79.03)

### Ablation Study

- **표현 스타일**: 도구 이름이나 위치 정보가 없을 때 성능이 하락하지만, 움직임 중심 학습을 한 모델은 이름 정보가 없어도 공간/동적 문맥을 통해 이를 보완하는 능력이 뛰어났다.
- **KFS 전략**: 균일 샘플링이나 코사인 유사도 기반 선택보다 제안된 KFS 방식이 훨씬 적은 수의 프레임($T'=8$)만으로도 최적의 성능에 도달함을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 수술 도구 분할에서 **"무엇처럼 보이는가(What it looks like)"**보다 **"어떻게 움직이는가(How it moves)"**가 훨씬 더 강력한 식별 신호가 될 수 있음을 입증하였다.

**강점**으로는 수술 도구의 명칭이 병원마다 다른 언어적 불일치 문제를 움직임이라는 물리적 특성으로 해결하여, 모델의 범용성과 일반화 능력을 극대화했다는 점이 꼽힌다. 또한, Key-frame Selection을 통해 긴 비디오에서도 효율적으로 중요한 시점만을 추출하여 연산 비용을 줄이면서 성능을 높였다.

**한계 및 논의 사항**으로는, 움직임이 거의 없는 정적인 상태의 도구를 식별해야 하는 상황에서는 여전히 외형 정보에 의존해야 한다는 점이 남아있다. 또한, 본 연구에서 사용한 $T'=8$이라는 하이퍼파라미터가 매우 복잡한 장기 움직임을 모두 포착하기에 충분한지에 대한 추가적인 분석이 필요할 수 있다.

## 📌 TL;DR

SurgRef는 수술 도구의 외형이 아닌 **동적인 움직임(Motion)**을 기반으로 도구를 식별하고 분할하는 새로운 프레임워크이다. 이를 위해 움직임 중심의 언어 표현이 담긴 **Ref-IMotion 데이터셋**을 구축하였으며, **Key-frame Selection** 모듈을 통해 효율적인 시간적 추론을 가능케 하였다. 결과적으로 외형의 유사성이나 용어의 비표준화 문제를 극복하고, 서로 다른 수술 절차 간에도 강력한 **Zero-shot 일반화 성능**을 보여줌으로써 차세대 지능형 수술 보조 시스템의 기반을 마련하였다.
