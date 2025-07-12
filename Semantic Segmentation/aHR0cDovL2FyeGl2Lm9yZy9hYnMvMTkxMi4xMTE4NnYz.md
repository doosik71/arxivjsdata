# A Comprehensive Analysis of Weakly-Supervised Semantic Segmentation in Different Image Domains
Lyndon Chan, Mahdi S. Hosseini, Konstantinos N. Plataniotis

## 🧩 Problem to Solve
픽셀 단위 주석 없이 이미지 레이블만을 사용하여 픽셀 클래스를 예측하는 약지도(Weakly-Supervised) 의미론적 분할(Semantic Segmentation, WSSS) 방법론은 주석 생성 비용을 크게 절감할 수 있어 실용성이 높습니다. 그러나 기존 WSSS 방법들은 주로 자연 환경 이미지(예: PASCAL VOC2012)에 최적화되어 개발되었으며, 이들 방법이 조직 병리(histopathology)나 위성 이미지와 같이 특성이 다른 도메인에 단순히 전이될 수 있는지, 그리고 좋은 성능을 유지할 수 있는지에 대한 연구는 부족합니다. 자연 환경 이미지는 배경과 전경 분리, 객체의 부분적 분할, 자주 함께 나타나는 객체 구분 등의 어려움을 겪는 반면, 조직 병리 및 위성 이미지는 모호한 경계, 클래스 동시 발생(class co-occurrence)과 같은 다른 유형의 문제에 직면합니다. 본 논문은 이러한 도메인 간의 성능 격차를 분석하고, 주어진 데이터셋에 가장 적합한 WSSS 방법을 결정하는 방법을 제시하고자 합니다.

## ✨ Key Contributions
*   **문헌 종합 검토:** 다중 클래스 의미론적 분할 데이터셋 및 이미지 레이블 기반 WSSS 방법론에 대한 포괄적인 문헌 검토를 제공합니다. 각 데이터셋 및 방법론의 특징과 해결하려는 문제를 설명합니다.
*   **광범위한 평가:** 자연 환경(PASCAL VOC2012), 조직 병리(ADP), 위성 이미지(DeepGlobe) 데이터셋에서 최신 WSSS 방법들(SEC, DSRG, IRNet, HistoSegNet)을 구현하고 정량적, 정성적으로 평가합니다.
*   **심층 분석 및 일반 원칙 제안:** 각 접근 방식이 다른 이미지 도메인 분할과 얼마나 잘 호환되는지 상세히 분석하고, WSSS를 다양한 이미지 도메인에 적용하기 위한 일반적인 원칙을 제안합니다. 특히, 분류 네트워크 큐의 희소성(sparsity), 자가지도 학습(self-supervised learning)의 이점, 높은 클래스 동시 발생 문제 해결 방안에 초점을 맞춥니다.

## 📎 Related Works
본 논문은 다양한 이미지 도메인에서의 다중 클래스 의미론적 분할 데이터셋과 약지도 의미론적 분할(WSSS) 방법론을 광범위하게 검토합니다.

*   **다중 클래스 의미론적 분할 데이터셋:**
    *   **자연 환경 이미지:** MSRC-21, SIFT Flow, PASCAL VOC2012, PASCAL-Context, COCO 2014, ADE20K, COCO-Stuff 등.
    *   **조직 병리 이미지:** C-Path, MMMP (H&E), HMT, NCT-CRC, ADP-morph/func 등.
    *   **가시광선 위성 이미지:** UC Merced Land Use, DeepGlobe Land Cover, EuroSAT Land Use 등.
    *   **도시 경관 이미지:** CamVid, CityScapes, Mapillary Vistas, BDD100K, ApolloScape 등.
*   **약지도 의미론적 분할 (WSSS):**
    *   **기대-최대화(Expectation-Maximization):** CCNN (Pathak et al., 2015), EM-Adapt (Papandreou et al., 2015).
    *   **다중 인스턴스 학습(Multiple Instance Learning, MIL):** MIL-FCN (Pathak et al., 2014), DCSM (Shimoda and Yanai, 2016), BFBP (Saleh et al., 2016), WILDCAT (Durand et al., 2017).
    *   **객체 제안 클래스 추론(Object Proposal Class Inference):** SPN (Kwak et al., 2017), PRM (Zhou et al., 2018).
    *   **자가지도 학습(Self-Supervised Learning):** SEC (Kolesnikov and Lampert, 2016b), MDC (Wei et al., 2018), AE-PSL (Wei et al., 2017), FickleNet (Lee et al., 2019), DSRG (Huang et al., 2018), PSA (Ahn and Kwak, 2018), IRNet (Ahn et al., 2019).
*   **위성 및 조직 병리 이미지를 위한 의미론적 분할 방법:**
    *   **위성 이미지:** DFCNet (Tian et al., 2018), Deep Aggregation Net (Kuo et al., 2018), FPN 변형 (Seferbekov et al., 2018) 등 완전지도 학습 방법이 주를 이루며, 약지도 학습으로는 Affinity-Net (PSA 변형) (Nivaggioli and Randrianarivo, 2019), SDSAE (Yao et al., 2016) 등이 있습니다.
    *   **조직 병리 이미지:** 슬라이딩 패치 기반 방법 (Ciresan et al., 2013), 슈퍼픽셀 기반 방법 (Xu et al., 2016), 완전 합성곱 네트워크(FCN) 기반 방법 (Chen et al., 2016) 등이 주로 이진 클래스 문제에 적용됩니다. 약지도 학습으로는 MCIL (Xu et al., 2014), EM-CNN (Hou et al., 2016), DWS-MIL (Jia et al., 2017), ScanNet (Lin et al., 2018), HistoSegNet (Chan et al., 2019) 등이 있습니다.

## 🛠️ Methodology
본 연구는 세 가지 대표적인 데이터셋에 대해 네 가지 최신 WSSS 방법을 비교 평가합니다.

*   **선정된 데이터셋:**
    *   **Atlas of Digital Pathology (ADP):** 조직 병리 이미지 데이터셋. 형태학적(morphological) 28개, 기능적(functional) 4개 유형으로 구성됩니다. `train` 세트를 학습에 사용하고, `tuning` 세트와 `segtest` 세트를 각각 검증 및 평가에 사용합니다.
    *   **PASCAL VOC2012:** 자연 환경 이미지 데이터셋. 20개 전경 클래스와 배경 클래스를 포함합니다. `trainaug` 세트를 학습에, `val` 세트를 평가에 사용합니다.
    *   **DeepGlobe Land Cover Classification:** 위성 이미지 데이터셋. 6개 지표면 클래스와 `unknown` 클래스를 포함합니다. `train` 세트의 75%를 학습에, 25%를 테스트에 사용합니다.
*   **선정된 WSSS 방법:**
    *   **SEC (Seed, Expand and Constrain):** (Kolesnikov and Lampert, 2016b)
        1.  분류 CNN 학습: 전경/배경 네트워크(VGG16 변형)를 이미지 레이블로 학습합니다.
        2.  CAM 생성: 학습된 CNN에서 클래스 활성화 맵(CAM)을 생성합니다.
        3.  시드(Seed) 생성: CAM을 임계값으로 처리하여 약한 지역화 큐(시드)를 생성하고 겹침을 해결합니다.
        4.  자가지도 FCN 학습: 생성된 시드를 의사(pseudo) GT로 사용하여 FCN(DeepLabv1)을 학습하며, 시딩 손실, 확장 손실, 제약 손실을 적용합니다. 추론 시 밀집 CRF(Dense CRF)로 후처리합니다.
    *   **DSRG (Deep Seeded Region Growing):** (Huang et al., 2018)
        1.  분류 CNN 학습: VGG16 기반 CNN을 학습합니다.
        2.  배경 활성화 생성: DRFI(Discriminative Regional Feature Integration) 방법으로 배경 활성화를 생성합니다.
        3.  시드 생성: 전경 CAM을 임계값으로 처리하여 컨볼루션 특징 기반 영역 확장(region growing) 시드를 생성합니다.
        4.  자가지도 FCN 학습: 영역 확장된 시드를 의사 GT로 사용하여 FCN(DeepLabv2)을 학습하며, 시딩 손실과 경계 손실을 적용합니다. 추론 시 밀집 CRF로 후처리합니다.
    *   **IRNet (Inter-pixel Relation Network):** (Ahn et al., 2019)
        1.  분류 CNN 학습 및 CAM 생성: ResNet50 기반 CNN을 학습하고 CAM을 생성합니다.
        2.  시드 생성: CAM을 임계값으로 처리하고 밀집 CRF로 정제하여 전경 시드를 생성합니다. 배경 시드는 CAM 신뢰도가 낮고 전경 시드가 없는 영역으로 정의됩니다.
        3.  자가지도 DF 및 CBM 학습: 백본 네트워크에서 변위 필드(Displacement Field, DF)와 클래스 경계 맵(Class Boundary Map, CBM)을 예측하는 두 가지 브랜치를 학습합니다.
        4.  CAM 무작위 워크(Random Walk) 전파: CBM의 역수를 전이 확률 행렬로 사용하여 CAM을 무작위 워크로 전파하여 클래스 경계 내에서 활성화를 확산시킵니다.
    *   **HistoSegNet:** (Chan et al., 2019)
        1.  분류 CNN 학습: VGG-16의 얕은 변형(X1.7)을 패치 수준 주석으로 학습합니다.
        2.  Grad-CAM 적용: 학습된 CNN에서 Grad-CAM을 사용하여 픽셀 수준 HTT(Histological Tissue Type) 예측을 추론합니다.
        3.  HTT 간 조정: 배경(`background`) 및 기타(`other`) 기능 클래스 활성화를 측정하고 클래스 간 활성화를 빼는 수작업 조정을 수행합니다.
        4.  밀집 CRF: 조정된 활성화 맵에 밀집 CRF를 적용하여 세분화된 픽셀 수준 분할 맵을 생성합니다.
*   **실험 설정:**
    *   모든 WSSS 방법은 VGG16 및 X1.7(M7) 아키텍처를 사용하여 총 8가지 네트워크-방법 조합으로 구현했습니다.
    *   시드 신뢰도 임계값은 경험적으로 조정했으며, 이미지 리사이징 및 증강(augmentation)을 적용했습니다.
    *   학습률 주기적 조정, ImageNet 사전 초기화를 사용했습니다.
    *   평가 지표는 평균 IoU (mean Intersection-over-Union, mIoU)를 사용했습니다:
        $$mIoU = \frac{1}{C} \sum_{c=1}^{C} \frac{|P_{c} \cap T_{c}|}{|P_{c} \cup T_{c}|}$$
        여기서 $P_{c}$는 클래스 $c$의 예측 픽셀이고, $T_{c}$는 클래스 $c$의 실제 픽셀입니다.
*   **절제 연구(Ablative Study):** HistoSegNet의 네트워크 아키텍처(VGG16의 8가지 변형)가 WSSS 성능에 미치는 영향을 분석했습니다. 더 얕은 네트워크(예: 3개 블록의 M3)가 특정 세분화 작업(특히 형태학적 유형)에 더 효과적임을 발견했으며, 이는 분류 성능이 높다고 반드시 세분화 성능이 좋다는 것을 의미하지 않음을 보여줍니다.

## 📊 Results
*   **ADP (조직 병리 데이터셋):**
    *   **정량적 성능:** HistoSegNet만이 Grad-CAM 베이스라인을 꾸준히 능가하며, ADP에 맞게 설계된 X1.7 네트워크가 VGG16보다 우수합니다. 자가지도 방법(SEC, DSRG, IRNet)은 성능이 좋지 않으며, SEC가 가장 나쁘고 DSRG가 그 다음, IRNet은 Grad-CAM과 비슷한 수준입니다.
    *   **정성적 성능:** X1.7 구성이 작은 세그먼트에 더 잘 대응하여 우수한 성능을 보였습니다. SEC와 DSRG는 객체 윤곽선은 잘 맞지만 객체 크기를 과장하는 경향이 있는 반면, HistoSegNet은 그렇지 않습니다.
*   **PASCAL VOC2012 (자연 환경 데이터셋):**
    *   **정량적 성능:** SEC와 DSRG만이 Grad-CAM 베이스라인을 일관되게 능가하며, SEC가 명확히 우수합니다. VGG16 네트워크가 M7 네트워크보다 전반적으로 우수합니다. M7 큐를 사용한 SEC가 전체적으로 가장 좋은 성능을 보였습니다. 본 논문의 결과는 원본 논문보다 다소 낮은데, 이는 배경 큐 생성 방법 차이 및 구현 세부 사항 때문일 수 있습니다.
    *   **정성적 성능:** VGG16 Grad-CAM은 M7보다 전체 객체를 더 잘 포착하여 VGG16 구성이 더 나은 성능을 보입니다. SEC와 DSRG는 Grad-CAM의 실수를 수정할 수 있지만, HistoSegNet은 종종 세그먼트를 잘못된 객체에 연결합니다. 모든 방법은 자주 함께 발생하는 객체(예: 보트와 물)를 구분하는 데 어려움을 겪습니다.
*   **DeepGlobe Land Cover Classification (위성 이미지 데이터셋):**
    *   **정량적 성능:** DSRG와 IRNet만이 Grad-CAM 베이스라인을 일관되게 능가하며 DSRG가 전반적으로 우수합니다. M7의 Grad-CAM이 열등함에도 불구하고 M7 네트워크가 VGG16보다 우수합니다. M7 큐를 사용한 DSRG가 전체적으로 가장 좋은 성능을 보였습니다. 완전지도 방법(DFCNet)에 비해서는 훨씬 낮은 성능을 보였습니다.
    *   **정성적 성능:** 모든 4가지 방법은 시각적으로 상당히 유사하게 예측하지만, DSRG와 HistoSegNet이 작은 세부 사항을 더 잘 포착합니다. VGG16은 M7보다 더 거친 예측을 합니다. 모든 방법은 `water` (물) 영역 분할에 어려움을 겪습니다.

## 🧠 Insights & Discussion
*   **분류 네트워크 큐 희소성의 영향:**
    *   분류 네트워크의 설계는 WSSS 성능에 중요한 영향을 미칩니다.
    *   Grad-CAM 세그먼트를 더 희소하게 생성하는 네트워크(예: X1.7/M7)는 실제 객체 인스턴스가 많은 데이터셋(예: ADP-func)에서 더 나은 성능을 보입니다. 반면, 더 적고 큰 세그먼트가 있는 데이터셋(예: VOC2012)에서는 VGG16과 같이 덜 희소한 큐를 생성하는 네트워크가 더 효과적입니다.
    *   새로운 데이터셋에 WSSS를 적용할 때, 실제 객체 인스턴스 수에 따라 적절한 희소성을 가진 분류 네트워크를 선택하는 것이 중요합니다. 이는 평균 5.22%의 mIoU 차이를 발생시킬 수 있습니다.
*   **자가지도 학습의 유익성:**
    *   현재 WSSS의 주요 접근 방식인 자가지도 학습(SEC, DSRG, IRNet)은 자연 환경 이미지에서는 효과적이지만, 조직 병리 이미지에서는 명확히 열등하며, 위성 이미지에서는 그 가치가 불분명합니다.
    *   자가지도 학습의 성능은 임계값 처리된 Grad-CAM 시드가 실제 세그먼트를 얼마나 잘 커버하는지(평균 재현율)에 크게 의존합니다.
    *   시드 재현율이 낮은 데이터셋(<40%, 예: VOC2012, DeepGlobe)에서는 자가지도 방법이 더 효과적입니다.
    *   시드 재현율이 높은 데이터셋(≥40%, 예: ADP-func, ADP-morph)에서는 HistoSegNet과 같이 자가지도 학습을 사용하지 않는 방법이 더 나은 성능을 보입니다.
    *   이는 자가지도 방법이 적은 시드로부터 광범위하게 예측하도록 학습되어, 시드가 이미 포괄적일 때는 오히려 성능에 해로울 수 있기 때문입니다.
    *   시드 임계값을 낮춰 시드 커버리지를 50% 이상으로 늘려도, SEC 및 DSRG 같은 자가지도 방법의 성능은 개선되지 않았습니다.
*   **높은 클래스 동시 발생 문제 해결:**
    *   이미지 레이블만으로 WSSS를 학습하는 것은 객체의 위치 정보를 제공하지 않으므로, 클래스 동시 발생이 잦은 데이터셋에서 특히 어렵습니다.
    *   DeepGlobe 데이터셋과 같이 클래스 동시 발생이 매우 높은 경우(50% 이상) `balancing`이라는 간단한 기술(가장 많은 클래스 레이블을 가진 학습 이미지를 절반 제거)을 통해 동시 발생을 줄이면 해당 클래스(예: `agriculture`, `forest`, `water`)의 성능이 향상됨을 보여주었습니다.
    *   이는 클래스 동시 발생이 WSSS의 중요한 도전 과제임을 시사하며, 더 효과적인 동시 발생 감소 방법이 필요함을 의미합니다.

## 📌 TL;DR
이미지 레이블 기반의 약지도 의미론적 분할(WSSS)은 주석 비용을 줄이지만, 자연 환경, 조직 병리, 위성 이미지 등 다양한 도메인에 걸쳐 일반화하기 어렵습니다. 본 논문은 SEC, DSRG, IRNet, HistoSegNet과 같은 최신 WSSS 방법들을 여러 도메인에서 종합적으로 평가했습니다. 주요 발견은 다음과 같습니다: (1) 각 방법은 개발된 도메인에서 최고의 성능을 보였습니다. (2) 분류 네트워크의 큐 희소성은 실제 세그먼트의 수에 따라 조절되어야 합니다. (3) 자가지도 학습은 초기 시드의 재현율이 낮을 때만 유용하며, 그렇지 않은 경우에는 자가지도 학습을 사용하지 않는 방법이 더 좋습니다. (4) 클래스 동시 발생은 WSSS 성능을 크게 저해하며, 이를 줄이는 간단한 데이터 균형 기법이 도움이 될 수 있습니다. 이는 다양한 이미지 도메인에 적용할 수 있는 더 일반화된 WSSS 방법론 개발이 필요함을 시사합니다.