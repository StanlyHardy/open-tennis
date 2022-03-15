from tqdm import tqdm


class Evaluator():
    def __init__(self):
        self.matching_name , self.matching_score, self.matching_serve =0, 0,0

    def evaluate(self,predicted: dict, gt_annotation: dict, total_frame_count = 0):
        if total_frame_count!= 0:
            for frame_num, prediction in tqdm(predicted.items(), desc="Evaluating..."):
                gt = gt_annotation[frame_num]

                gt_name_1, gt_name_2 = gt["name_1"].lower().strip(), gt["name_2"].lower().strip()
                pr_name_1, pr_name_2 = prediction["name_1"], prediction["name_2"]
                pr_score_1, pr_score_2 = prediction["score_1"], prediction["score_2"]
                gt_score_1, gt_score_2 = gt["score_1"].lower().strip(), gt["score_2"].lower().strip()
                gt_server , pred_server = gt["serving_player"],prediction["serving_player"]
                if gt_name_1 == pr_name_1 and \
                        gt_name_2 == pr_name_2:
                    self.matching_name = self.matching_name + 1

                if gt_score_1 == pr_score_1 and gt_score_2 == pr_score_2:
                    self.matching_score = self.matching_score + 1
                if gt_server == pred_server:
                    self.matching_serve = self.matching_serve+1

            print("Average correct Names: {} ".format(self.matching_name / total_frame_count))
            print("Average correct Scores: {} ".format(self.matching_score / total_frame_count))
            print("Average correct Serving Player: {} ".format(self.matching_serve / total_frame_count))
