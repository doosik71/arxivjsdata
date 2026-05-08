# Model Adaptation: Historical Contrastive Learning for Unsupervised Domain Adaptation without Source Data

Jiaxing Huang, Dayan Guan, Aoran Xiao, Shijian Lu

## 🧩 Problem to Solve

비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)은 레이블이 지정된 원본(Source) 도메인과 레이블이 없는 대상(Target) 도메인의 데이터 분포를 정렬하는 것을 목표로 합니다. 그러나 UDA는 훈련 중에 원본 데이터에 접근해야 하며, 이는 데이터 프라이버시, 데이터 이식성, 데이터 전송 효율성 측면에서 우려를 제기합니다.

본 연구는 원본 데이터에 접근하지 않고 원본 학습 모델을 대상 분포에 적응시키는 **비지도 모델 적응(Unsupervised Model Adaptation, UMA)** 또는 '원본 데이터 없는 비지도 도메인 적응'이라는 대안적 설정을 다룹니다. UMA 설정에서는 원본 학습 모델만이 유일한 정보이며, 레이블이 지정된 원본 데이터의 부재로 인해 도메인 적응이 훨씬 더 어렵고 모델 붕괴에 취약하다는 문제가 있습니다.

## ✨ Key Contributions

* **메모리 기반 학습 도입**: UMA를 위한 메모리 기반 학습을 최초로 탐구하여, 원본 가설(source hypothesis)을 유지하면서 레이블이 없는 대상 데이터를 위한 판별적(discriminative) 표현을 학습합니다.
* **혁신적인 Historical Contrastive Learning (HCL) 설계**:
  * **Historical Contrastive Instance Discrimination (HCID)**: 현재 모델과 과거 모델에서 생성된 임베딩을 대조하여 인스턴스 수준의 판별적 대상 표현을 학습하며, 원본 가설을 잊지 않도록 합니다.
  * **Historical Contrastive Category Discrimination (HCCD)**: 현재 및 과거 모델 간의 예측 일관성(prediction consistency)에 따라 의사 레이블(pseudo-label)에 가중치를 부여하여 범주 수준의 판별적 대상 표현을 학습합니다.
* **광범위한 실험**: HCL이 의미론적 분할, 객체 탐지, 이미지 분류 등 다양한 시각 작업 및 설정에서 최첨단 UMA 방법들보다 일관적으로 뛰어난 성능을 보임을 입증합니다.

## 📎 Related Works

* **비지도 모델 적응 (UMA) / Source-Free Domain Adaptation**: 원본 데이터에 접근하지 않고 원본 학습 모델을 적응시키는 연구로, [46, 47]은 분류를 위해 분류기를 고정하고 정보 최대화를 수행하고, [43]은 조건부 GAN을 사용하여 타겟과 유사한 스타일의 이미지를 생성하며, [44]는 객체 탐지를 위해 자체 엔트로피 감소를 제시합니다. [75]는 분할을 위한 타겟 예측의 불확실성을 줄이고, [49]는 데이터 없는 지식 증류(distillation)를 도입합니다.
* **도메인 적응 (UDA)**: 레이블이 지정된 원본 데이터가 필요한 일반적인 도메인 적응 방법론으로, 적대적 학습 [81, 53, 79], 자체 학습(self-training) [108, 72, 104], 이미지 변환 [26, 74, 45] 등이 있습니다.
* **메모리 기반 학습 (Memory-based Learning)**: 메모리 네트워크 [84], 시간적 앙상블 [42], Mean Teacher [77]와 같이 과거 가설/모델을 사용하여 현재 모델을 정규화하는 기법들이 있습니다.
* **대조 학습 (Contrastive Learning)**: 동일 인스턴스의 여러 뷰에서 판별적 표현을 학습하는 [85, 90, 22]와 같은 방법들이 있으며, HCL은 이를 UMA에 맞게 역사적 모델을 활용하도록 확장합니다.

## 🛠️ Methodology

제안하는 **Historical Contrastive Learning (HCL)**은 원본 데이터의 부재를 보완하기 위해 원본 가설을 기억하는 두 가지 핵심 설계로 구성됩니다.

1. **Historical Contrastive Instance Discrimination (HCID)**
    * **목표**: 원본 도메인 가설을 유지하면서 인스턴스 판별적 대상 표현을 학습합니다.
    * **과정**:
        * 현재 모델 $E_t$는 쿼리 샘플 $x_q$를 인코딩하여 $q_t = E_t(x_q)$를 생성합니다.
        * 과거 인코더 $E_{t-m}$는 키 샘플 $x_k^n$을 인코딩하여 $k_{t-m}^n = E_{t-m}(x_k^n)$를 생성합니다.
        * **손실 함수 ($L_{HisNCE}$)**: 쿼리 $q_t$를 긍정 키 $k_{t-m}^+$에 가깝게 당기고 모든 다른 (부정) 키에서 멀어지게 합니다.
        * $$ L_{HisNCE} = - \sum_{x_q \in X_{tgt}} \log \frac{\exp(q_t \cdot k_{t-m}^+ / \tau) r_{t-m}^+}{\sum_{i=0}^N \exp(q_t \cdot k_{t-m}^i / \tau) r_{t-m}^i} $$
        * 여기서 $\tau$는 온도 파라미터이며, $r$은 각 키의 신뢰도(reliability)를 나타냅니다 (분류 엔트로피로 추정). 이는 잘 학습된 과거 임베딩을 기억하고 그렇지 않은 임베딩의 영향을 줄입니다.
2. **Historical Contrastive Category Discrimination (HCCD)**
    * **목표**: 시각 인식 작업의 목표에 부합하는 범주 판별적 대상 표현을 학습합니다.
    * **과정**:
        * **Historical Contrastive Pseudo Label 생성**:
            * 현재 모델 $G_t$는 $x$에 대해 $p_t = G_t(x)$를 예측하고, 과거 모델 $G_{t-m}$는 $p_{t-m} = G_{t-m}(x)$를 예측합니다.
            * 의사 레이블 $\hat{y} = \Gamma(p_t)$ (표준 의사 레이블 생성 함수)를 계산합니다.
            * **역사적 일관성 ($h_{con}$)**: 현재 및 과거 모델 예측 간의 일관성을 측정합니다.
                $$ h_{con} = 1 - \text{Sigmoid}(||p_t - p_{t-m}||_1) $$
        * **손실 함수 ($L_{HisST}$)**: 가중치 교차 엔트로피 손실을 통해 대상 데이터 $x$에 대한 자체 학습을 수행합니다.
        * $$ L_{HisST} = - \sum_{x \in X_{tgt}} h_{con} \times \hat{y} \log p_x $$
        * 샘플에 대한 예측이 현재 모델과 과거 모델 간에 일관적이면 ($h_{con}$이 높으면), 해당 샘플의 자체 학습 손실에 대한 영향이 증가하고, 불일치하면 감소합니다.

**이론적 통찰**: HCID와 HCCD는 각각 특정 조건 하에서 기댓값 최대화(Expectation Maximization)를 통해 최적화되는 최대 우도(maximum likelihood) 문제로 모델링될 수 있으며 수렴성을 보장합니다.

## 📊 Results

* **의미론적 분할 (Semantic Segmentation)**: GTA5→Cityscapes 및 SYNTHIA→Cityscapes 작업에서 HCL은 최첨단 UMA 방법들(mIoU 기준)을 큰 폭으로 능가합니다. 기존 UMA 방법에 HCL을 통합하면 일관적으로 성능이 향상되며, 심지어 원본 데이터에 접근하는 최첨단 UDA 방법들과도 경쟁력 있는 성능을 달성합니다. HCID와 HCCD는 상호 보완적으로 작동하여 최상의 분할 결과를 생성합니다.
* **객체 탐지 (Object Detection)**: Cityscapes→Foggy Cityscapes 및 Cityscapes→BDD100k 작업에서 HCL은 최첨단 UMA 방법인 SFOD보다 명확하게 우수하며, UDA 방법들과도 경쟁력 있는 성능을 보입니다.
* **이미지 분류 (Image Classification)**: VisDA17 및 Office-31 벤치마크에서 HCL은 최첨단 UMA 방법들을 명확하게 능가하며, UDA 방법들과도 경쟁력 있는 성능을 보여줍니다.
* **시각 작업 전반의 일반화**: HCL은 의미론적 분할, 객체 탐지, 이미지 분류의 세 가지 대표적인 시각 작업에서 일관적으로 경쟁력 있는 성능을 달성하여 뛰어난 일반화 능력을 입증합니다.
* **상보성 연구**: HCL을 기존 UMA 방법과 결합하면 기존 방법의 성능이 일관적으로 향상됨을 보여줍니다.
* **특징 시각화 (t-SNE)**: HCL은 인스턴스 판별적이면서도 범주 판별적인 바람직한 특징 표현을 학습하며, 정성적으로 UR 및 SFDA보다 우수함을 보여줍니다.
* **학습 설정 전반의 일반화**: HCL은 부분 집합(partial-set) 적응 및 개방 집합(open-set) 적응 설정 모두에서 일관적으로 경쟁력 있는 성능을 달성합니다.
* **정성적 결과**: GTA5→Cityscapes 작업에서 HCL은 UR 및 SFDA보다 더 나은 분할 결과(예: 보도, 도로, 하늘)를 생성하며, 원본 가설을 효과적으로 보존합니다.

## 🧠 Insights & Discussion

* **원본 데이터 부재 문제 해결**: HCL은 과거 모델의 '역사적 원본 가설'을 활용하여 원본 데이터가 없는 UMA의 핵심적인 문제를 효과적으로 해결합니다. 이는 모델 붕괴를 방지하고 성능을 유지하는 데 결정적인 역할을 합니다.
* **상보적 설계의 시너지**: 인스턴스 수준의 판별(HCID)과 범주 수준의 판별(HCCD)이라는 두 가지 상보적인 설계가 결합되어 강력하고 포괄적인 감독 신호를 제공합니다. HCID는 미세한 인스턴스 구분을 통해 미확인 데이터에 대한 일반화 능력을 향상시키고, HCCD는 명확한 클래스 경계를 촉진하여 다운스트림 작업 목표와 잘 정렬되게 합니다.
* **적응적 지식 활용**: HCID에서 과거 임베딩의 신뢰도 점수를 사용하고 HCCD에서 의사 레이블에 대한 역사적 일관성을 사용하는 것은, 모델이 더 신뢰할 수 있고 안정적인 예측을 우선적으로 활용하여 학습된 지식을 적응적으로 활용할 수 있게 합니다.
* **UDA에 대한 경쟁력**: HCL이 원본 데이터에 접근할 수 있는 UDA 방법들과 유사하거나 심지어 더 나은 성능을 달성한다는 것은, 메모리 기반 적응의 잠재력과 효율성을 강력하게 시사합니다.
* **광범위한 적용 가능성**: 분할, 탐지, 분류 등 다양한 시각 작업과 부분 집합, 개방 집합 적응과 같은 여러 설정에서 효과를 입증함으로써, 제안된 접근 방식의 높은 일반성을 보여줍니다.

## 📌 TL;DR

**문제**: 레이블된 원본(Source) 데이터 없이 원본 모델을 대상(Target) 도메인에 적용하는 비지도 모델 적응(UMA)은 데이터 프라이버시, 이식성, 전송 효율성 문제로 인해 중요하지만, 원본 데이터의 부재로 인해 모델 붕괴에 취약합니다.
**방법**: 본 연구는 "Historical Contrastive Learning (HCL)"을 제안하여 과거 모델의 지식(historical source hypothesis)을 활용해 원본 데이터의 부재를 보완합니다. HCL은 두 가지 핵심 메커니즘으로 구성됩니다: (1) **Historical Contrastive Instance Discrimination (HCID)**은 현재 모델과 과거 모델의 임베딩을 대조하여 인스턴스 수준의 판별 학습을 수행하며, 과거 임베딩의 신뢰도를 평가하여 잘 학습된 지식만 기억하도록 가중치를 부여합니다. (2) **Historical Contrastive Category Discrimination (HCCD)**는 현재 모델과 과거 모델의 예측 일관성(historical consistency)에 따라 가중치를 부여한 의사 레이블(pseudo-label)을 생성하여 범주 수준의 판별 학습을 진행합니다.
**결과**: HCL은 다양한 시각 작업(의미론적 분할, 객체 탐지, 이미지 분류)과 적응 설정(부분집합, 개방집합 적응)에서 기존의 최첨단 UMA 방법들을 일관적으로 능가하며, 심지어 원본 데이터에 접근하는 UDA 방법들과도 경쟁력 있는 성능을 보여줍니다. 이는 HCL이 원본 데이터 없이도 원본 모델의 가설을 보존하면서 대상 도메인에 효과적으로 적응함을 입증합니다.
