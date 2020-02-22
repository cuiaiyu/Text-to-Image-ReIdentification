import json, os

HEAD = ['<!DOCTYPE html>','<html>', '<body>']
TAIL = ['</body>', '</html>']




class HTMLGenerator:
    def __init__(self, img_dir, anno_path):
        self.img_dir = img_dir
        with open(anno_path, 'r') as f:
            anns = json.load(f)
        self.anns = [ann for ann in anns if ann['split'] == 'val']
        self.id2anns = {ann['file_path']:ann for ann in self.anns}
        self.id2persons =  {ann['file_path']:ann['id'] for ann in self.anns}

    def _generate_head(self):
        out_string = '\n'.join(HEAD) + '\n'
        return out_string

    def _generate_tail(self):
        out_string = '\n'.join(TAIL) + '\n'
        return out_string
            

    def _generate_subject_section(self, exp_name='default', epoch=0, subset='Full'):
        out_string = '<h1>Wider Person Search by Language - Visualization </h1>' + '\n'
        out_string += '<h2>Exp Meta</h2>' + '\n'
        out_string += '<p>Exp Name: %s</p>' % exp_name + '\n'
        out_string += '<p>Epoch: %d</p>' % epoch + '\n'
        out_string += '<p>Subset: %s </p>' % subset + '\n'
        return out_string

    def _generate_retrieval_table(self, submission_fn, K=100):
        # head
        out_string = '<h2>Retrieval Result</h2>' + '\n'
        out_string += '<table>' + '\n'
        out_string += '<tr>' + '\n'
        out_string += '<th>Query Index</th>' + '\n'
        out_string += '<th>Query Caption</th>' + '\n'
        out_string += '\n'.join(['<th> R%d </th>' % i for i in range(1, 11)]) + '\n'
        out_string += '</tr>' + '\n'
        # body
        with open(submission_fn, 'r') as f:
            all_results = f.readlines()
        K = len(all_results) if K == -1 else K
        for i, result in enumerate(all_results[:K]):
            curr_line = result[:-1] if result.endswith('\n') else result
            query, recalls = curr_line.split()
            query_id = self.id2persons[query]
            out_string += '<tr>' + '\n'
            out_string += '<td> %d </td>' % i + '\n'

            # query
            query_cap = self.id2anns[query]['captions'][0]
            out_string += '<td> %s </td>' % query_cap + '\n'

            # recalls
            for recall in recalls.split(','):
                recall_id = self.id2persons[recall]
                img_path = os.path.join(self.img_dir, recall)
                if query_id == recall_id:
                    out_string += '<td><img src="%s" height=128 wight=48" style="border:solid; border-color:red;"/></td>' % img_path + '\n'
                else:
                    out_string += '<td><img src="%s" height=128 wight=48" /></td>' % img_path + '\n'
            out_string += '</tr>' + '\n'
        f.close()
        return out_string

    def _generate_retrieval_table_false_only(self, submission_fn, K=-1):
        # head
        out_string = '<h2> Retrieval Result </h2>' + '\n'
        out_string += '<table>' + '\n'
        out_string += '<tr>' + '\n'
        out_string += '<th> Query Index </th>' + '\n'
        out_string += '<th> Query Caption </th>' + '\n'
        out_string += '\n'.join(['<th> R%d </th>' % i for i in range(1, 11)]) + '\n'
        out_string += '</tr>' + '\n'
        # body
        with open(submission_fn, 'r') as f:
            all_results = f.readlines()
        K = len(all_results) if K == -1 else K
        for i, result in enumerate(all_results[:K]):
            curr_line = result[:-1] if result.endswith('\n') else result
            query, recalls = curr_line.split()
            query_id = self.id2persons[query]
            recalls = recalls.split(',')
            recall_ids = [self.id2persons[recall] == query_id for recall in recalls]
            if sum(recall_ids) > 0:
                continue
            out_string += '<tr>' + '\n'
            out_string += '<td> %d </td>' % i + '\n'

            # query
            query_cap = self.id2anns[query]['captions'][i%2]
            out_string += '<td> %s </td>' % query_cap + '\n'

            # recalls
            for recall in recalls:
                img_path = os.path.join(self.img_dir, recall)
                out_string += '<td> <img src="%s" height=128 wight=48" /> </td>' % img_path + '\n'
            out_string += '</tr>' + '\n'
        return out_string


    def generate(self, submission_fn, K=100, false_only=False, out_path='tmp.html'):
        out_string = self._generate_head()
        #print(out_string)
        subset_title = 'false_only_{} (if True, only those queries with R@10 = 0 displayed.)'
        out_string += self._generate_subject_section(subset=subset_title.format(false_only))
        if false_only:
            out_string += self._generate_retrieval_table_false_only(submission_fn, -1)
        else:
            out_string += self._generate_retrieval_table(submission_fn, K)
        out_string += self._generate_tail()
        with open(out_path, 'w') as f:
            f.write(out_string)


if __name__ == '__main__':
    data_root = '/shared/rsaas/aiyucui2/wider_person/wider/val1/'
    Generator = HTMLGenerator(img_dir='img', anno_path=data_root+'val1_anns.json')
    Generator.generate(submission_fn='../tmp.txt', false_only=True, K=1000, out_path='../../http/false_only.html')



