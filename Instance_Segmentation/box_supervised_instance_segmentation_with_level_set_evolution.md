# Box-supervised Instance Segmentation with Level Set Evolution

Wentong Li, Wenyu Liu, Jianke Zhu, Miaomiao Cui, Xiansheng Hua, and Lei Zhang

## 🧩 Problem to Solve

픽셀 단위 마스크 주석(pixel-wise mask annotations)에 크게 의존하는 기존 인스턴스 분할(Instance Segmentation) 방법은 레이블링 비용이 매우 비쌉니다. 본 논문은 이러한 높은 비용 문제를 해결하기 위해, 간단한 경계 상자(bounding box) 주석만을 사용하여 인스턴스 분할을 수행하는 **상자-지도 인스턴스 분할(box-supervised instance segmentation)** 분야의 성능을 향상시키는 것을 목표로 합니다.

## ✨ Key Contributions

* 고전적인 레벨 셋 모델(level set model)과 딥러닝을 통합하여 새로운 레벨 셋 진화 기반 상자-지도 인스턴스 분할 접근 방식을 제안했습니다. 이는 상자-지도 인스턴스 분할 문제를 다루는 최초의 딥 레벨 셋 기반 방법입니다.
* 바운딩 박스 영역 내에서 강력한 레벨 셋 진화(level set evolution)를 달성하기 위해 저수준 이미지 특징과 고수준 딥 구조적 특징(deep structural features)을 통합했으며, 초기 레벨 셋 설정을 위해 박스 투영 함수(box projection function)를 활용했습니다.
* COCO, Pascal VOC, 원격 감지 데이터셋 iSAID, 의료 영상 데이터셋 LiTS를 포함한 네 가지 도전적인 벤치마크에서 상자-지도 인스턴스 분할 분야의 새로운 최첨단 성능을 달성했습니다.

## 📎 Related Works

* **상자-지도 인스턴스 분할 (Box-supervised Instance Segmentation)**: 픽셀 수준 마스크 주석 대신 바운딩 박스 주석만 사용하는 방법들. Mask R-CNN 기반의 MIL(Multiple Instance Learning) 문제로 접근한 BBTP [16], 색상 쌍 유사성(color-pairwise affinity)을 활용한 BoxInst [44], 프록시 마스크 레이블(proxy mask labels) 생성에 초점을 맞춘 BBAM [26] 및 DiscoBox [25] 등이 있습니다. 이들은 일반적으로 복잡한 파이프라인이나 노이즈에 취약한 문제가 있었습니다.
* **레벨 셋 기반 분할 (Level Set-based Segmentation)**: 이미지 분할을 연속적인 에너지 최소화 문제로 공식화하는 고전적인 변이 접근 방식. Mumford-Shah [36] 및 Chan-Vese [7] 모델이 대표적입니다. 최근에는 Levelset R-CNN [15], DVIS-700 [54] 등 딥 네트워크에 레벨 셋을 포함시키는 연구가 진행되었으나, 대부분 픽셀 단위 마스크의 완전한 지도를 전제로 했습니다. 본 논문은 이러한 기존 레벨 셋 기반 방법과 달리, 상자 주석만을 사용한 약한 지도(weak supervision) 방식으로 레벨 셋 진화를 수행합니다.

## 🛠️ Methodology

본 논문은 고전적인 Chan-Vese 에너지 기반 레벨 셋 모델을 딥 뉴럴 네트워크에 통합하여 상자-지도 인스턴스 분할을 수행하는 새로운 방법을 제안합니다.

1. **레벨 셋 진화 (Level Set Evolution)**:
    * Mask R-CNN 계열의 SOLOv2 [48] 모델을 사용하여 인스턴스 인식 마스크 맵($M$)을 예측하고, 이를 각 인스턴스에 대한 레벨 셋 함수 $\phi(x,y)$로 간주합니다.
    * 레벨 셋 진화는 주어진 바운딩 박스 $B$ 내에서 이루어집니다.
    * 다음과 같은 변형된 Chan-Vese 에너지 함수 $F(\phi, I, c_{1}, c_{2}, B)$를 최소화하여 객체 경계를 학습합니다.
        $$F(\phi,I,c_{1},c_{2},B) = \int_{\Omega \in B} |I^{\ast}(x,y)-c_{1}|^{2}\sigma(\phi(x,y))dxdy$$
        $$+ \int_{\Omega \in B} |I^{\ast}(x,y)-c_{2}|^{2}(1-\sigma(\phi(x,y)))dxdy + \gamma \int_{\Omega \in B} |\nabla \sigma(\phi(x,y))|dxdy$$
        여기서 $I^{\ast}(x,y)$는 정규화된 입력 이미지 또는 고수준 특징이며, $\sigma$는 시그모이드(sigmoid) 함수입니다. $c_1$과 $c_2$는 각각 경계 내부와 외부 픽셀 값의 평균입니다.
    * 위 에너지 함수는 완전 미분 가능하며, 경사 하강법을 통해 $\phi$를 반복적으로 업데이트하여 최적의 경계를 찾습니다: $\phi_{i} = \phi_{i-1} + \Delta t \frac{\partial \phi_{i-1}}{\partial t}$.

2. **입력 데이터 항 (Input Data Terms)**:
    * 레벨 셋 진화를 더욱 강력하게 만들기 위해, 저수준 입력 이미지 ($I_{\text{u}}$)뿐만 아니라 SOLOv2의 마스크 특징($F_{\text{mask}}$)에서 추출된 고수준 딥 구조적 특징 ($I_{\text{f}}$)도 활용합니다.
    * $I_{\text{f}}$는 장거리 의존성(long-range dependencies)을 모델링하고 객체 구조를 보존하기 위해 트리 필터(tree filter) [27, 41]로 강화됩니다.
    * 최종 에너지 함수는 두 가지 특징을 결합합니다: $F(\phi) = \lambda_{1} \ast F(\phi,I_{\text{u}},c^{\text{u}}_{1},c^{\text{u}}_{2},B) + \lambda_{2} \ast F(\phi,I_{\text{f}},c^{\text{f}}_{1},c^{\text{f}}_{2},B)$.

3. **레벨 셋 초기화 (Level Set Initialization)**:
    * 레벨 셋의 초기화는 중요하며, 수동 라벨링에 의존하는 기존 방식과 달리, 본 논문은 박스 투영 함수(box projection function) [44]를 사용하여 각 단계에서 초기 레벨 셋 $\phi_{0}$의 대략적인 추정치를 자동으로 생성합니다.
    * 이는 예측된 마스크 맵과 ground-truth 박스 간의 x축, y축 투영 차이를 1D Dice 계수 [35]로 측정하여 계산합니다: $F(\phi_{0})_{\text{box}} = P_{\text{dice}}(m^{\text{x}}_{\text{p}},m^{\text{x}}_{\text{b}}) + P_{\text{dice}}(m^{\text{y}}_{\text{p}},m^{\text{y}}_{\text{b}})$.

4. **손실 함수 (Loss Function)**:
    * 전체 손실 함수 $L$은 범주 분류를 위한 Focal Loss ($L_{\text{cate}}$)와 인스턴스 분할을 위한 레벨 셋 에너지 ($L_{\text{inst}}$)로 구성됩니다: $L = L_{\text{cate}} + L_{\text{inst}}$.
    * $L_{\text{inst}}$는 레벨 셋 에너지 $F(\phi)$와 박스 투영 함수 $F(\phi_{0})_{\text{box}}$를 결합합니다:
        $$L_{\text{inst}} = \frac{1}{N_{\text{pos}}} \sum_{k} \mathbb{1}\{p^{\ast}_{i,j}>0\} \{F(\phi) + \alpha F(\phi_{0})_{\text{box}}\}$$

5. **추론 (Inference)**:
    * 레벨 셋 진화는 학습 과정에서만 네트워크 최적화를 위한 암묵적인 지도(implicit supervisions)로 사용됩니다.
    * 추론 시에는 기존 SOLOv2 네트워크와 동일하게 입력 이미지에서 마스크 예측이 직접 생성됩니다. 추가적인 컨볼루션 레이어 하나만 사용하여 고수준 특징을 생성하므로 오버헤드가 적습니다.

## 📊 Results

* **Pascal VOC**: BoxInst [44]보다 ResNet-50 및 ResNet-101 백본에서 각각 2.0%, 1.8% AP 성능이 우수하며, DiscoBox [25]보다 AP$_{50}$에서 4.1% 높게 나타나 최상위 성능을 달성했습니다. 특히, AP$_{75}$에서 BoxInst [44]와 DiscoBox [25]를 각각 1.7%, 1.2% 능가하여 정확한 경계 분할에서 강점을 보였습니다.
* **COCO**: 동일한 백본에서 BBTP [16]보다 12.3% AP, BBAM [26]보다 7.7% AP, BoxCaseg [47]보다 2.5% AP 우수한 성능을 보였습니다. BoxInst [44]보다 ResNet-101 및 ResNet-101-DCN 백본에서 각각 0.2%, 0.4% AP 높게 나타났습니다. 특히 대형 객체(AP$_{L}$)에서 BoxInst [44]보다 2.4% AP 높은 성능을 달성했습니다. 소형 객체(AP$_{S}$)에서는 약간 낮은 경향을 보였습니다.
* **Deep Variational Methods 비교**: 마스크 전체 지도(fully mask-supervised) 방식인 DeepSnake [38] 및 Levelset R-CNN [15]보다 뛰어난 성능을 보이며, DVIS-700 [54]과 비슷한 경쟁력 있는 결과를 달성하여 약한 지도 방식과 완전 지도 방식 간의 성능 격차를 줄였습니다.
* **원격 감지 (iSAID) 및 의료 영상 (LiTS)**: iSAID에서 BoxInst [44]보다 2.3% AP, LiTS에서 3.8% AP 높은 성능을 보여, 복잡한 배경이나 밀집된 객체 등 다양한 시나리오에서 강건한 성능을 입증했습니다.

## 🧠 Insights & Discussion

* 본 연구는 고전적인 레벨 셋 모델을 딥러닝 프레임워크에 성공적으로 통합하여 상자-지도 인스턴스 분할의 성능을 크게 향상시켰습니다. 이는 픽셀 단위 마스크 주석 없이도 높은 품질의 분할 마스크를 얻을 수 있음을 보여줍니다.
* 특히, 저수준 입력 이미지 특징과 고수준 딥 구조적 특징을 모두 활용하여 레벨 셋 진화를 유도하고, 바운딩 박스 제약을 통해 불필요한 노이즈 간섭을 줄여 강건한 성능을 달성했습니다.
* 원격 감지 및 의료 영상과 같이 객체가 밀집되어 있거나 배경과 전경이 유사한 도전적인 환경에서도 기존의 픽셀 관계 모델 기반 방법들보다 뛰어난 강건성을 보였습니다. 이는 레벨 셋 기반의 곡선 진화가 객체 경계를 효과적으로 추출함을 시사합니다.
* 다만, 소형 객체의 경우 바운딩 박스 내에서 전경과 배경을 구분할 풍부한 특징이 부족하여 레벨 셋 진화에 어려움이 있을 수 있다는 한계가 있습니다.
* 훈련 스케줄이 길수록 레벨 셋 진화가 더 나은 수렴을 달성하여 성능이 크게 향상되는 것으로 나타났으며, 트리 필터(tree filter)를 사용하여 딥 구조적 특징을 강화하는 것이 성능 개선에 기여함을 입증했습니다.

## 📌 TL;DR

본 논문은 픽셀 마스크 주석 없이 바운딩 박스 주석만으로 인스턴스 분할을 수행하는 **상자-지도 인스턴스 분할** 문제 해결을 위해, 고전적인 **레벨 셋 진화 모델**을 딥 뉴럴 네트워크와 통합한 새로운 단일 샷(single-shot) 접근 방식을 제안합니다. 제안된 방법은 SOLOv2 기반으로 인스턴스 마스크 맵을 레벨 셋으로 활용하고, 원본 이미지와 고수준 딥 특징을 입력으로 받아 완전 미분 가능한 Chan-Vese 에너지 함수를 반복적으로 최소화하여 객체 경계를 최적화합니다. 경계 상자 투영 함수로 레벨 셋 초기화를 자동화하고, 바운딩 박스 내에서 진화를 제한하여 노이즈 간섭을 줄입니다. 광범위한 실험 결과, 제안된 방법은 Pascal VOC, COCO, iSAID(원격 감지), LiTS(의료 영상) 등 다양한 데이터셋에서 기존 상자-지도 인스턴스 분할 방법 대비 최첨단 성능을 달성하며, 마스크 전체 지도 방식과의 성능 격차를 성공적으로 줄였습니다.
