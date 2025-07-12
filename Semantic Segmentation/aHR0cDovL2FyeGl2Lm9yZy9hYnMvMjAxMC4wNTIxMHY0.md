# Generalized Few-shot Semantic Segmentation
Zhuotao Tian, Xin Lai, Li Jiang, Shu Liu, Michelle Shu, Hengshuang Zhao, Jiaya Jia

## 🧩 Problem to Solve
시맨틱 분할(Semantic Segmentation) 모델 훈련은 방대한 양의 정밀하게 주석된 데이터가 필요하여, 새로운 클래스에 빠르게 적응하기 어렵습니다. 기존의 Few-Shot Segmentation (FS-Seg)은 소수의 예시만으로 새로운 클래스를 분할하는 문제를 다루지만, 다음과 같은 제약 사항이 있습니다:
1.  **지원 샘플 의존성**: 쿼리 샘플에 존재하는 클래스를 지원 샘플이 반드시 포함해야 합니다. 이는 실제 환경에서 사용자에게 사전에 필요한 클래스에 대한 지식을 요구하는 비현실적인 가정을 만듭니다.
2.  **클래스 평가 범위 제한**: FS-Seg는 새로운 클래스(Novel Classes)만 평가하며, 실제 시맨틱 분할 모델은 기본 클래스(Base Classes)와 새로운 클래스 모두를 동시에 처리해야 합니다.
이러한 제약으로 인해 기존 SOTA(State-of-the-Art) FS-Seg 모델들은 실제 환경에서 기본 클래스와 새로운 클래스를 동시에 분할하는 능력(일반화 능력)이 부족합니다.

## ✨ Key Contributions
*   **GFS-Seg 벤치마크 제안**: 기존 FS-Seg를 확장하여 "Generalized Few-Shot Semantic Segmentation (GFS-Seg)"이라는 새로운 벤치마크를 제안합니다. GFS-Seg는 소수의 예시만 있는 새로운 카테고리와 충분한 예시가 있는 기본 카테고리를 동시에 분할하는 모델의 일반화 능력을 분석합니다.
*   **FS-Seg 모델의 한계 분석**: GFS-Seg 설정에서 기존 대표적인 FS-Seg 모델들이 크게 뒤떨어지며, 이러한 성능 차이가 FS-Seg의 제한적인 설정에서 비롯됨을 최초로 보여줍니다.
*   **Context-Aware Prototype Learning (CAPL) 제안**: GFS-Seg 문제를 해결하기 위해 컨텍스트 인식 프로토타입 학습(CAPL) 방법을 제안합니다. CAPL은 다음 두 가지 방식으로 성능을 크게 향상시킵니다:
    1.  지원 샘플에서 공동 발생(co-occurrence) 사전 지식 활용.
    2.  각 쿼리 이미지의 내용에 따라 분류기에 컨텍스트 정보를 동적으로 풍부하게 부여.
*   **높은 일반화 능력**: CAPL은 기존 시맨틱 분할 모델(예: FCN, PSPNet, DeepLab)에 구조적 변경 없이 적용할 수 있으며, FS-Seg 설정에서도 경쟁력 있는 성능을 달성합니다.

## 📎 Related Works
*   **시맨틱 분할(Semantic Segmentation)**: FCN, Encoder-Decoder 구조(SegNet, U-Net), Dilated Convolution(DeepLab), 컨텍스트 모델링(Global/Pyramid Pooling), 어텐션 모델(CCNet, DANet) 등의 발전을 통해 성능 향상을 이루었지만, 충분한 주석 데이터 없이는 새로운 클래스에 쉽게 적응할 수 없습니다.
*   **소수 학습(Few-shot Learning)**: 소수의 레이블된 예시만으로 새로운 클래스를 예측하는 것을 목표로 합니다. 메타 학습(Meta-learning) 기반 및 거리 학습(Metric-learning) 기반 방법론이 주를 이루며, 데이터 증강을 통해 과적합을 방지하기도 합니다. 이미지 분류에서는 일반화된 소수 학습이 탐구되었지만, 시맨틱 분할은 픽셀 단위 레이블링과 컨텍스트 정보의 중요성에서 차이가 있습니다.
*   **소수 분할(Few-shot Segmentation, FS-Seg)**: OSLSM에 의해 처음 소개된 이 설정은 소수의 지원 샘플로 새로운 클래스에 대한 픽셀 단위 레이블링을 수행합니다. 프로토타입 학습(PL, PANet)이나 사전 지식 활용(PFENet) 등의 아이디어가 적용되었습니다. 그러나 본 논문의 5장에서 보여주듯이, 쿼리 이미지에 포함된 타겟 클래스에 대한 사전 지식 없이 기본 클래스와 새로운 클래스를 모두 포함하는 실용적인 설정에서는 기존 FS-Seg 모델들이 제대로 작동하지 않습니다.

## 🛠️ Methodology
GFS-Seg는 크게 세 단계로 구성됩니다:
1.  **기본 클래스 학습 단계(Base Class Learning Phase)**: 충분한 레이블된 기본 클래스 데이터로 모델을 훈련하여 좋은 특징 표현을 학습합니다.
2.  **새로운 클래스 등록 단계(Novel Class Registration Phase)**: 제한된 K개의 레이블된 샘플을 통해 N개의 새로운 클래스 정보를 획득하고 새로운 분류기(classifier)를 구성합니다.
3.  **평가 단계(Evaluation Phase)**: 기본 클래스와 새로운 클래스($C_{b} \cup C_{n}$) 모두에 대한 레이블을 예측합니다. 쿼리 이미지에 어떤 클래스가 포함되어 있는지에 대한 사전 지식은 주어지지 않습니다.

**GFS-Seg를 위한 기준 모델(Baseline)**:
*   기존 시맨틱 분할 모델의 특징 추출기(Feature Extractor)와 분류기(Classifier)를 사용합니다.
*   기본 클래스 분류기 가중치($P_{b} \in R^{N_{b} \times d}$)를 기본 클래스 학습 단계에서 역전파를 통해 학습된 기본 프로토타입으로 간주합니다.
*   새로운 클래스 프로토타입($P_{n} \in R^{N_{n} \times d}$)은 식 (1)과 같이 K개의 지원 샘플에 대해 특징 추출 후 마스크 평균 풀링(Mask Average Pooling)하여 생성합니다.
    $$p_{i} = \frac{1}{K} \sum_{j=1}^{K} \frac{\sum_{h,w} [m_{ij} \circ F(s_{ij})]_{h,w}}{\sum_{h,w} [m_{ij}]_{h,w}}$$
*   $P_{b}$와 $P_{n}$을 연결하여 새로운 분류기 $P_{all}$을 형성하며, 쿼리 이미지 픽셀에 대한 예측은 식 (2)와 같이 코사인 유사도($\phi$)를 기반으로 수행합니다.
    $$O_{x,y} = \arg \max_{i} \frac{\exp(\alpha\phi(F(q_{x,y}),p_{i}))}{\sum_{p_{i} \in P_{all}} \exp(\alpha\phi(F(q_{x,y}),p_{i}))}$$

**Context-Aware Prototype Learning (CAPL)**:
분류기에 컨텍스트 정보를 효과적으로 통합하여 GFS-Seg의 성능을 향상시킵니다.
*   **지원 컨텍스트 강화(Support Contextual Enrichment, SCE)**:
    *   "새로운 클래스 등록 단계"에서 지원 샘플 내 기본 클래스들의 공동 발생 컨텍스트를 활용합니다.
    *   지원 샘플 내 기본 클래스 마스크로부터 생성된 새로운 기본 프로토타입 $p_{b,i}^{sup}$ (식 3)와 원래 분류기의 기본 프로토타입 $p_{b,i}^{cls}$의 가중 합으로 업데이트된 프로토타입 $p_{b,i}$를 생성합니다 (식 4).
        $$p_{b,i} = \gamma_{i}^{sup} \cdot p_{b,i}^{cls} + (1-\gamma_{i}^{sup}) \cdot p_{b,i}^{sup}$$
    *   적응형 가중치 $\gamma_{i}^{sup}$는 $G^{sup}(p_{b,i}^{cls}, p_{b,i}^{sup})$ 함수(MLP 사용)를 통해 데이터에 따라 동적으로 결정됩니다.
*   **동적 쿼리 컨텍스트 강화(Dynamic Query Contextual Enrichment, DQCE)**:
    *   "평가 단계"에서 개별 쿼리 이미지의 컨텍스트에 분류기를 동적으로 적응시킵니다.
    *   원래 분류기의 임시 예측($y_{qry}$)을 통해 쿼리 이미지에서 기본 클래스의 카테고리 대표($p_{b}^{qry}$)를 생성합니다 (식 5).
        $$p_{b}^{qry} = \text{Softmax}(y_{qry}^{t}) \times F(q)$$
    *   이 $p_{b}^{qry}$와 원래 분류기의 기본 프로토타입 $p_{b,i}^{cls}$를 가중 합하여 동적으로 강화된 프로토타입 $p_{b,i}^{dyn}$을 얻습니다 (식 6).
        $$p_{b,i}^{dyn} = \gamma_{i}^{qry} \cdot p_{b,i}^{cls} + (1-\gamma_{i}^{qry}) \cdot p_{b,i}^{qry}$$
    *   가중치 $\gamma_{i}^{qry}$는 $G^{qry}(p_{b,i}^{cls}, p_{b,i}^{qry})$ 함수(코사인 유사도 사용)를 통해 신뢰도를 측정하여 결정됩니다.
*   **최종 분류기**: SCE와 DQCE를 통해 얻은 강화된 기본 클래스 프로토타입 $P_{b}^{capl}$ (식 7)를 새로운 클래스 프로토타입 $P_{n}$과 결합하여 최종 분류기 $P_{all}^{capl}$을 형성합니다.
    $$P_{b}^{capl} = P_{b} + P_{b}^{dyn}$$
*   **훈련**: 특징 추출기가 분류기 가중치와 호환되는 특징을 생성하도록 훈련 방식을 수정합니다. "가짜 지원(Fake Support)" 샘플을 무작위로 선택하여 "가짜 새로운 클래스(Fake Novel Classes)"와 "가짜 컨텍스트 클래스(Fake Context Classes)"로 나누어 실제 상황을 모방하고, 이에 따라 프로토타입을 업데이트합니다. 최종적으로는 $P_{b}^{capl}$을 사용하여 계산된 표준 교차 엔트로피 손실을 최소화합니다.

## 📊 Results
*   **GFS-Seg에서 FS-Seg 모델과의 비교**:
    *   CANet, PFENet, SCL, PANet 등 기존 SOTA FS-Seg 모델들은 GFS-Seg 설정(기본 및 새로운 클래스 동시 식별)에서 총 mIoU가 크게 낮아 좋지 않은 성능을 보였습니다. 이는 그들의 제한된 훈련/평가 스키마 때문입니다.
    *   반면, CAPL이 적용된 모델(PANet + CAPL, DeepLab-V3 + CAPL, PSPNet + CAPL)은 GFS-Seg 설정에서 기존 FS-Seg 모델을 크게 능가하는 우수한 성능을 달성했습니다. 예를 들어, PANet + CAPL은 1-shot에서 PANet의 26.97%에서 51.60%로 총 mIoU를 향상시켰습니다.
*   **세분화 연구(Ablation Study)**:
    *   SCE와 DQCE 모두 CAPL의 성능 향상에 필수적인 요소임을 입증했습니다 (표 2).
    *   $\gamma_{qry}$에 코사인 유사도를, $\gamma_{sup}$에 MLP를 사용하는 것이 가장 효과적이었습니다.
    *   CAPL의 훈련 전략과 컨텍스트 강화 전략은 서로 보완적이며, 둘 중 하나라도 없으면 성능이 저하됨을 확인했습니다 (표 3).
*   **FS-Seg에 CAPL 적용**:
    *   CAPL은 PANet 및 PFENet의 프로토타입 구성 프로세스를 변경하여, 기존 FS-Seg 설정에서도 기준 모델에 상당한 성능 향상을 가져왔습니다 (표 4).
    *   특히 COCO-20i 데이터셋에서 HSNet과 같은 SOTA 모델과 경쟁하거나 능가하는 결과를 보였습니다.

## 🧠 Insights & Discussion
*   **GFS-Seg의 실용성**: GFS-Seg는 실제 응용 시나리오에 더 가깝습니다. 사용자가 쿼리 이미지에 어떤 클래스가 있는지 사전에 알지 못하며, 기본 클래스와 새로운 클래스를 동시에 예측해야 하는 경우에 대한 해결책을 제시합니다.
*   **기존 FS-Seg의 한계**: 기존 FS-Seg 모델은 에피소드 방식의 훈련/테스트 스키마로 인해 배경과 특정 새로운 클래스 간의 이진 분할에만 초점을 맞춥니다. 또한, 많은 모델이 훈련 중 백본을 고정하여 다중 클래스 라벨링이 필요한 GFS-Seg와 같은 복잡한 시나리오에 대한 적응력이 떨어집니다.
*   **CAPL의 유효성**: CAPL은 지원 샘플과 쿼리 샘플 모두에서 컨텍스트 단서를 동적으로 활용하여 이 문제를 해결합니다. 적응형 가중치 체계를 통해 원래의 학습된 지식과 새로운 컨텍스트 정보를 균형 있게 조절합니다.
*   **제한 사항**: CAPL은 동적 컨텍스트 강화에 중점을 두지만, FS-Seg에서 쿼리 및 지원 특징 간의 밀집된 공간 추론(dense spatial reasoning)을 위한 새로운 설계를 도입하지 않습니다. 따라서 Pascal-5i와 같은 특정 FS-Seg 설정에서는 HSNet과 같은 하이퍼-상관관계(hyper-correlations)를 활용하는 고급 방법에 비해 성능이 약간 뒤처질 수 있습니다. 그러나 의미론적 단서가 더 중요한 COCO-20i와 같은 도전적인 데이터셋에서는 뛰어난 성능을 보입니다.

## 📌 TL;DR
**문제**: 기존 Few-Shot Semantic Segmentation (FS-Seg)은 소수의 예시만으로 새로운 클래스를 분할하지만, 실제 환경에서는 지원 샘플에 대한 사전 지식 없이 기본 클래스와 새로운 클래스를 동시에 분할해야 하는 제약이 있습니다.
**제안 방법**: 본 논문은 더 실용적인 벤치마크인 Generalized Few-Shot Semantic Segmentation (GFS-Seg)을 제안하고, 이를 해결하기 위해 Context-Aware Prototype Learning (CAPL)을 개발했습니다. CAPL은 지원 샘플의 공동 발생 사전 지식과 쿼리 이미지의 내용에 기반한 동적 컨텍스트 정보를 활용하여 분류기를 강화합니다.
**주요 결과**: 실험 결과, 기존 SOTA FS-Seg 모델은 GFS-Seg 설정에서 성능이 크게 저하되는 반면, CAPL은 GFS-Seg에서 뛰어난 성능을 보이며 기존 FS-Seg 벤치마크에서도 경쟁력 있는 결과를 달성합니다.