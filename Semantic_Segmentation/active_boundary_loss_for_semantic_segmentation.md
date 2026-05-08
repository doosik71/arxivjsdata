# Active Boundary Loss for Semantic Segmentation

Chi Wang, Yunke Zhang, Miaomiao Cui, Peiran Ren, Yin Yang, Xuansong Xie, Xian-Sheng Hua, Hujun Bao, Weiwei Xu

## 🧩 Problem to Solve

시맨틱 분할(Semantic Segmentation)은 종종 미세한 객체 경계 세부 정보가 부족하고 흐릿한 결과를 생성합니다. 이는 주로 완전 합성곱 네트워크(FCN)의 다운샘플링 작업이 객체 경계 전반에 걸쳐 특징을 부드럽게 만들고 컨텍스트 정보를 과도하게 전파하기 때문입니다. 기존의 경계 인식(boundary-aware) 방법들은 경험적인 성공에도 불구하고, 특히 작고 얇은 객체에 대한 경계에서 여전히 상당한 오류를 보입니다. 일반적으로 사용되는 교차 엔트로피(Cross-Entropy, CE) 손실은 픽셀 수준 분류에 초점을 맞춰 경계 정렬을 명시적으로 강제하지 않으며, IoU(Intersection-over-Union) 손실은 전체 영역에 더 주목하므로 경계 일치에 집중하지 못합니다. 따라서 시맨틱 분할 결과의 품질을 향상시키기 위해 시맨틱 분할과 경계 감지 간의 상호 의존성을 추가로 연구해야 할 필요가 있습니다.

## ✨ Key Contributions

* 시맨틱 분할을 위한 새로운 활성 경계 손실(Active Boundary Loss, ABL)을 제안합니다.
* ABL은 예측된 경계(Predicted Boundaries, PDB)와 실제 경계(Ground-Truth Boundaries, GTB) 간의 정렬을 종단 간 훈련(end-to-end training) 동안 점진적으로 촉진합니다.
* 경계 정렬 문제를 미분 가능한 방향 벡터 예측 문제로 공식화하여 각 반복에서 예측된 경계의 이동을 안내합니다.
* 모델에 구애받지 않아(model-agnostic) 기존 분할 네트워크 훈련에 쉽게 통합되어 경계 세부 정보를 개선할 수 있습니다.
* 경계 이동 시 발생할 수 있는 충돌을 억제하기 위해 그래디언트 흐름 제어(detaching operation)를 제안하며, 이는 ABL 성공에 결정적인 역할을 합니다.
* 도전적인 이미지 및 비디오 객체 분할 데이터셋에서 ABL을 사용한 훈련이 경계 F-점수(F-score)와 평균 IoU(mIoU)를 효과적으로 개선함을 입증합니다.
* ABL은 고전적인 활성 윤곽선(active contour) 방법의 변형으로 볼 수 있으며, 동적으로 PDB를 GTB로 이동시킵니다.

## 📎 Related Works

* **FCN 기반 시맨틱 분할:** FCN(Long et al. 2015), U-Net(Ronneberger et al. 2015), DeeplabV3(Chen et al. 2017a), OCR(Yuan et al. 2020a), SwinTransformer(Liu et al. 2021) 등 인코더-디코더 구조를 활용하여 픽셀별 레이블링을 생성하는 방법들. VOS(Video Object Segmentation)에는 STM(Oh et al. 2019) 등이 있습니다.
* **경계 인식 시맨틱 분할:**
  * **멀티태스크 훈련:** 시맨틱 경계를 감지하기 위한 추가 브랜치를 삽입하는 방법(Chen et al. 2020; Takikawa et al. 2019).
  * **정보 흐름 제어:** 경계를 통한 정보 흐름 제어에 중점을 두어 픽셀 간의 특징 차이를 유지하는 방법(Bertasius et al. 2016).
  * **사후 정제(Post-refinement):** DenseCRF(Krähenbühl et al. 2011)나 Segfix(Yuan et al. 2020b)처럼 분할 결과의 경계 주위를 정제하는 방법. 그러나 이들은 얇은 객체의 경계를 잘 처리하지 못할 수 있습니다.
* **가장 관련성 높은 연구:** Boundary Loss(BL, Kervadec et al. 2019)는 불균형 이진 분할을 위해 설계된 지역 IoU 손실입니다. ABL은 BL과 달리 PDB 픽셀에 초점을 맞춰 더 나은 정렬을 달성하며, IoU 손실(Lovász-softmax)과 함께 사용됩니다.

## 🛠️ Methodology

ABL은 PDB의 변화를 지속적으로 모니터링하여 가능한 이동 방향을 결정하며, 계산은 크게 두 단계로 나뉩니다.

1. **Phase I: 다음 후보 경계 픽셀 결정**
    * **PDB 감지:** 현재 네트워크 출력인 클래스 확률 맵 $P \in \mathbb{R}^{C \times H \times W}$에서 PDB를 감지합니다. 각 픽셀 $i$에 대해 $P_i$와 2-이웃 픽셀 $P_j$ 사이의 KL-다이버전스($KL(P_i, P_j)$)를 계산하여 경계 맵 $B$를 생성합니다. $B_i=1$은 PDB가 존재함을 나타냅니다. 고정된 임계값 대신 입력 이미지 총 픽셀의 1% 미만이 경계 픽셀이 되도록 적응형 임계값을 사용합니다.
    * **대상 방향 맵 계산:** PDB 상의 픽셀 $i$에 대해, GTB로부터 가장 가까운 이웃 픽셀 $j$를 다음 후보 경계 픽셀로 선택합니다. GTB는 픽셀 $i$와 $j$의 실제 클래스 레이블이 같은지 확인하여 결정합니다. 이 $j$의 위치는 $i$에 대한 오프셋으로 변환되어 8차원 원-핫 벡터 $D_g_i$로 인코딩됩니다. PDB의 이동을 가속화하기 위해 $B$를 1픽셀 팽창(dilate)시킨 영역에서 이 작업을 수행합니다.

2. **Phase II: 손실 공식화**
    * **예측 방향 맵 계산:** 픽셀 $i$와 그 8-이웃 픽셀 $P_{i+\Delta_k}$ 사이의 KL-다이버전스($KL(P_i, P_{i+\Delta_k})$)를 로짓(logits)으로 사용하여 8차원 벡터 $D_p_i$를 계산합니다. 이는 픽셀 $i$의 클래스 확률 분포와 $j$의 KL-다이버전스를 증가시키고, 나머지 이웃 픽셀들과의 KL-다이버전스를 감소시키도록 합니다.
    * **ABL 계산:** PDB 상의 픽셀에 대해 ABL은 가중치 교차 엔트로피 손실로 계산됩니다:
        $$ ABL = \frac{1}{N_b} \sum_{i} \Lambda(M_i) CE(D_p_i, D_g_i) $$
        여기서 $N_b$는 PDB 상의 픽셀 수이고, $\Lambda(x) = \frac{\min(x, \theta)}{\theta}$는 GTB까지의 가장 가까운 거리 $M_i$를 사용하여 가중치를 부여합니다($\theta=20$). $M_i=0$인 픽셀(이미 GTB에 있는 픽셀)은 ABL 계산에서 제외됩니다.

* **충돌 억제(Conflict Suppression):**
  * **Detaching Operation:** Pytorch의 detaching 연산을 통해 ABL의 그래디언트는 PDB 상의 픽셀에 대해서만 계산되며, 이웃 픽셀에 대해서는 계산되지 않습니다. 이는 모순적인 그래디언트 흐름을 차단하여 성능 저하를 방지합니다.
  * **레이블 스무딩(Label Smoothing):** 원-핫 목표 확률 분포의 가장 큰 확률을 0.8로, 나머지를 0.2/7로 설정하여 ABL을 정규화합니다. 이는 네트워크 파라미터 업데이트 시 과도하게 확신하는 결정을 피하도록 돕습니다.

* **전체 훈련 손실:**
    $$ L_t = CE + IoU + w_a ABL $$
    여기서 $CE$는 교차 엔트로피 손실, $IoU$는 Lovász-softmax 손실(작은 객체가 무시되지 않도록 돕고 초기 훈련 단계의 노이즈를 균형 있게 조절)이며, $w_a$는 ABL의 가중치입니다.

## 📊 Results

* **데이터셋 및 지표:** Cityscapes(19개 클래스), ADE20K(150개 클래스) 및 DAVIS-2016(VOS) 데이터셋에서 픽셀 정확도(pixAcc), 평균 IoU(mIoU), 경계 F-점수(1, 3, 5픽셀 팽창)를 사용하여 평가했습니다.
* **손실 항 영향:** ABL을 CE+IoU에 추가하면 Cityscapes에서 mIoU가 0.3%p, ADE20K에서 0.65%p 향상되었습니다. ABL은 클래스 수가 많은 데이터셋(ADE20K)에서 더 큰 기여를 보였습니다.
* **Detaching 연산의 중요성:** detaching 연산 없이 훈련할 경우 mIoU가 약 3%p 감소하는 등 성능이 크게 저하되어, 충돌 억제가 ABL의 성공에 필수적임을 입증했습니다.
* **FKL(Full KL-divergence) 손실과의 비교:** 모든 경계에 KL-다이버전스 손실을 적용하는 FKL은 PDB에 집중하는 ABL만큼 성능이 좋지 않았습니다. 이는 ABL이 PDB 픽셀에 더 많은 주의를 기울여 네트워크 동작을 점진적으로 조정하기 때문입니다.
* **기존 방법과의 비교:**
  * **Segfix:** DeepLabV3 및 OCR 네트워크에서 Segfix와 유사하거나 더 나은 mIoU 성능을 달성했습니다. 특히, 3픽셀 및 5픽셀 경계 F-점수에서는 ABL이 Segfix보다 우수했으며, 얇은 객체(예: 신호등)의 경계를 더 잘 처리했습니다. Segfix는 후처리 방식인 반면 ABL은 종단 간 훈련 손실입니다.
  * **Boundary Loss (BL):** 의료 영상 데이터셋(WMH)에서 ABL+GDL(Generalized Dice Loss)이 BL+GDL보다 더 높은 Dice 유사 계수(DSC)와 더 작은 Hausdorff 거리(HD)를 달성했습니다. ABL이 PDB 픽셀에 초점을 맞춰 더 나은 정렬을 이루는 반면, BL은 GTB 근처 픽셀의 영향력을 약화시키는 경향이 있습니다.
* **비디오 객체 분할(VOS) 결과:** STM 네트워크를 IABL로 미세 조정했을 때 DAVIS-2016 유효성 검사 세트에서 J-평균(region similarity)은 약 0.7%p, F-평균(contour accuracy)은 약 1%p 향상되었습니다.
* **정성적 결과:** 시티스케이프 및 ADE20K 이미지, DAVIS-2016 비디오에서 ABL을 통해 훈련된 모델이 PDB를 GTB로 점진적으로 정제하고 시맨틱 경계 세부 정보를 크게 개선함을 시각적으로 보여주었습니다.

## 🧠 Insights & Discussion

ABL의 핵심은 예측된 경계와 실제 경계 간의 **동적인 정렬**을 점진적으로 유도한다는 점입니다. 이는 기존의 교차 엔트로피나 IoU 손실이 경계 세부 사항을 명시적으로 다루지 못하는 한계를 보완합니다. 특히 **detaching 연산**을 통한 그래디언트 흐름 제어는 모순적인 그래디언트 문제를 해결하여 ABL의 안정적인 훈련과 성능 향상에 결정적인 역할을 합니다.

ABL은 모델에 구애받지 않아 다양한 CNN 및 Transformer 기반 네트워크에 쉽게 통합될 수 있으며, 이미지뿐만 아니라 비디오 객체 분할에서도 경계 품질을 개선하는 데 효과적임을 입증했습니다. 또한, 전체 이미지에 걸쳐 손실을 적용하는 것보다 PDB 픽셀에 집중하는 ABL의 방식이 더 효과적임을 보여, 경계 영역에 대한 집중적인 학습의 중요성을 강조합니다.

얇은 객체나 복잡한 경계에 대한 강점은 ABL이 실제 시나리오에서 섬세한 객체 분할 문제를 해결하는 데 유리함을 시사합니다. 미래 연구로는 손실 내에서 충돌을 추가로 줄이는 방법과 경계 인식 손실을 깊이 예측(depth prediction)과 같은 다른 컴퓨터 비전 태스크에 적용하는 방안이 논의될 수 있습니다.

## 📌 TL;DR

* **문제:** 시맨틱 분할은 종종 흐릿한 경계를 생성하며, 기존 손실 함수는 경계 정렬을 명시적으로 다루지 않습니다.
* **제안 방법:** `Active Boundary Loss (ABL)`는 예측된 경계(PDB)가 실제 경계(GTB)를 향해 점진적으로 움직이도록 유도합니다. 이는 경계 정렬을 미분 가능한 방향 예측 문제로 공식화하고, 충돌을 방지하기 위한 그래디언트 detaching 연산을 포함합니다. ABL은 교차 엔트로피 및 IoU 손실과 결합되어 사용됩니다.
* **주요 결과:** ABL은 이미지 및 비디오 시맨틱 분할에서 경계 F-점수와 mIoU를 크게 향상시키며, 얇은 객체 경계 처리에서 특히 효과적입니다. 또한, 이는 기존의 경계 인식 방법들보다 우수하거나 동등한 성능을 보입니다.
