import arxiv

serach = arxiv.Search(query='quantum', max_results=10, sort_by=arxiv.SortCriterion.SubmittedDate)
x0 = list(serach.results())
x1 = x0[0]
x1.authors
x1.doi
x1.categories
x1.updated
x1.title
x1.primary_category
x1.entry_id
# x1.download_pdf(filename='tbd00.pdf')

search = arxiv.Search(id_list=['1605.08386'])
x0 = list(search.results())


client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
search = arxiv.Search(id_list=['1605.08386'])
x0 = list(client.results(search))
