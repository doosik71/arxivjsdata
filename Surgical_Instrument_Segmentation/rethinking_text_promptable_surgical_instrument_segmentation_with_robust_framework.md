# Rethinking Text-Promptable Surgical Instrument Segmentation with Robust Framework

Tae-Min Choi, Juyoun Park (2025)

## 🧩 Problem to Solve

본 논문은 수술 도구 분할(Surgical Instrument Segmentation, SIS) 분야에서 최근 도입된 Text-promptable segmentation 방식들이 가진 치명적인 한계점을 해결하고자 한다. 기존의 promptable 방식들은 텍스트 프롬프트로 입력된 도구가 이미지 내에 반드시 존재한다는 가정(Oracle information) 하에 작동한다. 하지만 실제 수술 환경에서는 특정 도구가 화면에 나타날지 여부가 불확실하며, 존재하지 않는 도구에 대해 프롬프트가 주어졌을 때 모델이 강제로 마스크를 생성하여 False Positive(오탐)를 발생시키는 문제가 있다.

수술 환경에서 이러한 오탐은 매우 위험하며, 기존의 평가 프로토콜 또한 정답(Ground Truth)을 기반으로 존재하는 도구에 대해서만 프롬프트를 입력하는 방식으로 진행되어 실제 환경의 불확실성을 반영하지 못하고 있다. 따라서 본 연구의 목표는 도구의 존재 여부를 알 수 없는 상태에서도 모든 후보 카테고리에 대해 프롬프트를 입력하고, 모델이 스스로 존재 여부를 판단하여 정확한 마스크를 생성하게 하는 **Robust text-promptable Surgical Instrument Segmentation (R-SIS)** 과업을 정의하고 이를 해결하는 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같이 요약할 수 있다.

1. **R-SIS 과업 정의**: 도구의 존재 여부에 대한 사전 정보 없이 모든 후보 클래스에 대해 프롬프트를 제공하고, 모델이 시각적 특징만을 기반으로 존재 여부를 판단하여 분할을 수행하는 새로운 벤치마크 및 평가 프로토콜을 제안하였다.
2. **RoSIS 프레임워크 설계**: 존재 예측(Existence prediction) 브랜치와 다중 프롬프트 타입, 그리고 반복적 정제(Iterative refinement) 전략을 통합한 RoSIS(Robust Surgical Instrument Segmentation) 모델을 개발하였다.
3. **강건한 분할 성능 증명**: EndoVis2017 및 2018 데이터셋을 통해 기존의 Vision-based 및 Promptable 모델들과 비교하여, 특히 False Positive를 획기적으로 줄이면서도 높은 분할 정확도를 유지함을 입증하였다.

## 📎 Related Works

### 1. Surgical Instrument Segmentation (SIS)

기존의 SIS 연구들은 주로 이미지 기반의 segmentation 방법(TernausNet, ISINet 등)이나 비디오의 시간적 일관성을 이용하는 방법(TraSeTR, MATIS 등)에 집중해 왔다. 최근에는 SAM(Segment Anything Model)을 미세 조정하거나 CLIP과 같은 Vision-Language Model(VLM)을 활용하여 텍스트 프롬프트 기반의 분할을 시도하는 TP-SIS, SurgicalSAM 등의 연구가 등장하였다.

### 2. Referring Image Segmentation (RIS)

RIS는 텍스트 표현에 기반하여 객체를 분할하는 과업이다. LAVT와 같은 최신 모델들은 ViT와 BERT를 결합하여 높은 성능을 보이지만, 기본적으로 "텍스트가 가리키는 객체가 이미지에 존재한다"는 가정을 전제로 한다. 최근 RefSegformer와 같은 연구에서 객체가 없는 'Empty-target' 상황을 다루기 시작했으며, 본 논문은 이러한 Robust RIS의 개념을 수술 도구 분할 영역으로 확장하여 적용하였다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

RoSIS는 Encoder-Decoder 구조를 기본으로 하며, 시각적 특징과 텍스트 특징을 조기에 융합하고, 최종적으로 도구의 존재 확률($p^c$)과 세그멘테이션 마스크($M^c$)를 동시에 예측한다.

- **Encoder**: 이미지 인코더로 **Swin Transformer**를, 텍스트 인코더로 **BERT**를 사용한다.
- **Multi-Modal Fusion Block (MMFB)**: 시각적 특징($v$)과 언어 특징($l$)을 융합하기 위해 두 개의 Multi-Head Cross Attention(MHCA) 모듈을 사용한다. 첫 번째 MHCA는 언어 토큰을 통해 언어 특징을 정제하고, 두 번째 MHCA는 이를 시각적 특징과 상호작용시킨다.
- **Selective Gate Block (SGB)**: 융합된 특징이 시각적 정보보다 언어 정보에 지나치게 지배되지 않도록 제어하는 게이팅 메커니즘이다.
- **Decoder**: Multi-scale deformable attention pixel decoder와 FPN 구조를 사용하여 정밀한 마스크를 생성한다.

### 2. 존재 예측 브랜치 및 손실 함수

디코더와 병렬로 존재 확률 $p^c$를 계산하는 브랜치가 존재한다. 이 브랜치는 디코더의 출력과 BERT의 raw language feature를 MHCA로 융합하여 최종 존재 확률을 도출한다. 학습을 위한 손실 함수는 다음과 같이 정의된다.

$$L = BCELoss(p^c, y^c) + \lambda \cdot CELoss(M^c, M_{gt}^c)$$

여기서 $y^c$는 존재 여부의 정답, $M_{gt}^c$는 정답 마스크이며, $\lambda$는 마스크 손실의 가중치를 조절하는 하이퍼파라미터이다.

### 3. 프롬프트 생성 전략

모델의 이해도를 높이기 위해 세 가지 타입의 프롬프트를 사용한다.

- **Class Name Prompt**: "The {class name}" 형태의 기본 이름.
- **Visual Description Prompt**: GPT-4를 이용해 생성한 도구의 외관 특징 묘사.
- **Location Prompt**: "The {class name} on the {location}" 형태로, 도구가 주로 나타나는 4개 사분면(left-top, left-bottom, right-top, right-bottom) 정보를 포함한다.

### 4. 추론 단계의 반복적 정제 (Iterative Refinement)

추론 시에는 정답 위치 정보를 알 수 없으므로 다음과 같은 2단계 과정을 거친다.

- **Iteration 1**: 이름 프롬프트와 GPT-4 묘사 프롬프트를 입력하여 초기 마스크($M_1^c, M_2^c$)와 존재 확률($p_1^c, p_2^c$)을 구한다. 평균 확률이 0.5 이상인 경우에만 다음 단계로 진행한다.
- **Iteration 2**: 1단계에서 예측된 마스크의 무게 중심(Center of Mass)을 계산하여 위치 정보를 파악하고, 이를 기반으로 위치 프롬프트를 생성하여 다시 입력한다. 이를 통해 정제된 마스크 $M_3^c$와 확률 $p_3^c$를 얻는다.

최종 마스크 $M^c$는 다음과 같은 조건부 수식으로 결정된다.

$$M^c = \begin{cases} 0, & \text{if } \frac{p_1^c + p_2^c}{2} < 0.5 \\ \frac{M_1^c + M_2^c}{2}, & \text{if } \frac{p_1^c + p_2^c}{2} \geq 0.5 \text{ and } p_3^c < 0.5 \\ \frac{M_1^c + M_2^c + M_3^c}{3}, & \text{if } \frac{p_1^c + p_2^c}{2} \geq 0.5 \text{ and } p_3^c \geq 0.5 \end{cases}$$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: EndoVis2017 및 EndoVis2018.
- **비교 대상**: Vision-based 모델(TernausNet, ISINet, S3Net 등) 및 Promptable 모델(LAVT, TP-SIS, RefSegformer).
- **평가 지표**: Ch IoU(존재 클래스 IoU), ISI IoU(전체 클래스 IoU), mc IoU(평균 클래스 IoU), False Positive Rate(FPR), Precision, Recall, F1-score.

### 2. 정량적 결과

- **분할 성능**: EndoVis2018 데이터셋에서 RoSIS는 Ch IoU 77.66%, ISI IoU 76.16%, mc IoU 44.54%를 기록하며 모든 Promptable 모델과 대부분의 Vision-based 모델을 압도하였다.
- **강건성(Robustness)**: Table 5 결과에 따르면, RoSIS는 기존 promptable 모델들에 비해 FPR을 획기적으로 낮추었다. (예: EV18에서 LAVT 0.6155 $\rightarrow$ RoSIS 0.0892). 이는 존재 예측 모듈이 불필요한 마스크 생성을 효과적으로 억제하고 있음을 보여준다.
- **일반화 능력**: Cross-dataset 실험(EV18 $\rightarrow$ EV17 및 그 반대)에서도 RoSIS는 다른 promptable 모델보다 훨씬 적은 Ch IoU와 ISI IoU의 격차를 보이며 높은 일반화 성능을 입증하였다.

### 3. Ablation Study

- **프롬프트 및 정제 전략**: 이름 프롬프트만 사용했을 때보다 GPT-4 프롬프트와 반복적 정제(Iterative Refinement)를 함께 사용했을 때 ISI IoU가 크게 상승하였다.
- **구조적 기여**: MMFB, SGB, Raw Language features, Language Token을 순차적으로 추가했을 때 성능이 계단식으로 향상됨을 확인하여, 각 모듈의 유효성을 검증하였다.

## 🧠 Insights & Discussion

본 논문은 단순히 모델의 성능을 높이는 것이 아니라, **"평가 프로토콜의 현실성"**이라는 근본적인 문제를 제기하였다. 기존 promptable SIS 모델들이 보여준 높은 수치는 사실상 '정답지(Oracle information)'를 미리 보고 프롬프트를 구성했기 때문이라는 점을 지적하며, R-SIS라는 더 엄격한 기준을 제시한 점이 매우 가치 있다.

**강점**:

- 존재 예측 브랜치를 통해 False Positive 문제를 해결하여 실제 수술 시스템에 적용 가능한 수준의 신뢰성을 확보하였다.
- 단순한 아키텍처 변경보다는 프롬프트 설계와 추론 전략(반복적 정제)을 통해 성능을 끌어올린 점이 효율적이다.

**한계 및 논의**:

- 반복적 정제 과정에서 1단계 예측이 완전히 틀렸을 경우, 2단계의 위치 프롬프트가 오히려 잘못된 가이드를 제공할 가능성이 있다.
- 현재는 정해진 클래스들에 대해서만 동작하며, 완전히 새로운(unseen) 도구에 대한 제로샷(Zero-shot) 확장성에 대한 심층적인 분석은 부족하다.

## 📌 TL;DR

이 논문은 수술 도구 분할에서 텍스트 프롬프트 기반 모델들이 가지는 **"도구 존재 가정"**의 오류를 지적하고, 이를 해결하기 위한 **R-SIS(Robust text-promptable SIS)** 과업과 **RoSIS** 프레임워크를 제안한다. RoSIS는 존재 예측 모듈과 반복적 정제 전략을 통해 **False Positive를 획기적으로 줄이면서도 높은 분할 정확도**를 달성하였으며, 이는 실제 수술 로봇 시스템의 자동화 및 인터랙티브 시스템 구축에 있어 매우 중요한 기반 기술이 될 것으로 기대된다.
