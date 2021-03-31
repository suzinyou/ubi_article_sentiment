from setuptools import setup
from article_sentiment.kobert import __version__


setup(name='article_sentiment',
      version=__version__,
      url='https://github.com/suzinyou/ubi_article_sentiment',
      license='Apache-2.0',
      author='Suzin You & Heewon Jeon',
      author_email='suzinyou.sy@gmail.com',
      description='Article sentiment classification with Korean BERT pre-trained cased (KoBERT) ',
      packages=['article_sentiment', 'article_sentiment.model', 'article_sentiment.data', 'article_sentiment.kobert'],
      long_description=open('README.md', encoding='utf-8').read(),
      zip_safe=False,
      include_package_data=True,
      )