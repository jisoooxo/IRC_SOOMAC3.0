from setuptools import find_packages, setup

package_name = 'soomac_irc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={package_name: ['templates/*.html', 'static/*.js', 'static/ingredients/*.webp', 'nest.proto']},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='roma',
    maintainer_email='badukjoo@naver.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'stt_node = soomac_irc.stt_node:main',
            'stt_nemotron_node = soomac_irc.stt_nemotron_node:main',
            'tts_node = soomac_irc.tts_node:main',
            'llm_node = soomac_irc.llm_node:main',
            'llm_debug = soomac_irc.llm_debug:main',
            'ui_node = soomac_irc.ui_node:main',
        ],
    },
)
